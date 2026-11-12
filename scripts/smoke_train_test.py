"""Tiny real training/held-out evaluation on synthetic images and Betti features.

This checks numerical execution and parameter updates, not paper performance.
Run each model in a separate process to keep framework memory bounded.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def make_split(count, seed):
    import cv2
    import numpy as np
    from BettiVectorization import generate_betti_vectors

    rng = np.random.default_rng(seed)
    images, labels = [], np.arange(count, dtype=np.int64) % 2
    for label in labels:
        image = np.full((32, 32, 3), rng.integers(220, 256), dtype=np.uint8)
        x, y = rng.integers(13, 19, size=2)
        color = (int(rng.integers(0, 35)),) * 3
        cv2.circle(image, (int(x), int(y)), 9, color, thickness=-1)
        if label == 1:
            cv2.circle(image, (int(x), int(y)), 4, tuple(int(v) for v in image[0, 0]), thickness=-1)
        images.append(image)
    features = generate_betti_vectors(images, n_jobs=1).to_numpy(dtype=np.float32)
    return np.stack(images), features, labels


def prepare_data():
    import numpy as np

    train_images, train_ph, train_y = make_split(4, 17)
    test_images, test_ph, test_y = make_split(2, 29)
    assert not any(np.array_equal(a, b) for a in train_images for b in test_images)
    mean = train_ph.mean(axis=0)
    std = np.maximum(train_ph.std(axis=0), 1.0)
    return train_images, (train_ph - mean) / std, train_y, test_images, (test_ph - mean) / std, test_y


def train_vit(data, steps):
    import torch
    import cv2
    import numpy as np
    from TopoViT import TopoViT

    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = TopoViT(
        num_classes=2, ph_in_dim=16, ph_token_len=25, ph_embed_dim=32,
        ph_num_heads=4, tf_num_heads=8, tf_layers=1, dropout=0.0,
        pretrained=False,
    )
    train_images, train_ph, train_y, test_images, test_ph, test_y = data

    def image_tensor(images):
        rgb = np.stack([cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2RGB) for im in images])
        values = rgb.astype(np.float32) / 255.0
        values = (values - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(values.transpose(0, 3, 1, 2).copy())

    image_train, image_test = image_tensor(train_images), image_tensor(test_images)
    ph_train, ph_test = torch.from_numpy(train_ph), torch.from_numpy(test_ph)
    y_train, y_test = torch.from_numpy(train_y), torch.from_numpy(test_y)
    tracked = {
        "image_branch": model.vit.patch_embed.proj.weight,
        "topology_branch": model.PH_embed.weight,
        "classifier": model.classifier.weight,
    }
    before = {name: value.detach().clone() for name, value in tracked.items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for step in range(steps):
        start = (step % 2) * 2
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_train[start:start + 2], ph_train[start:start + 2])
        loss = torch.nn.functional.cross_entropy(logits, y_train[start:start + 2])
        assert torch.isfinite(loss), "Non-finite training loss"
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(float(loss.detach()))
    updated = {name: not torch.equal(before[name], value.detach()) for name, value in tracked.items()}
    assert all(updated.values()), updated
    model.eval()
    with torch.no_grad():
        logits = model(image_test, ph_test)
        test_loss = torch.nn.functional.cross_entropy(logits, y_test)
        assert logits.shape == (2, 2) and torch.isfinite(logits).all() and torch.isfinite(test_loss)
        accuracy = float((logits.argmax(dim=1) == y_test).float().mean())
    return {"model": "TopoViT", "train_losses": losses, "test_loss": float(test_loss), "test_accuracy": accuracy, "updated": updated, "torch": torch.__version__}


def train_cnn(data, steps):
    import tensorflow as tf
    import cv2
    import numpy as np
    from topo_cnn import build_cnn_tda_model

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(7)
    model = build_cnn_tda_model((32, 32, 3), 400, 2, base_model_name="MobileNetV2", weights=None)
    train_images, train_ph, train_y, test_images, test_ph, test_y = data

    def image_array(images):
        rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]).astype(np.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(rgb)

    image_train, image_test = image_array(train_images), image_array(test_images)
    classifier = model.layers[-1].kernel
    before = classifier.numpy().copy()
    losses = []
    for step in range(steps):
        start = (step % 2) * 2
        model.reset_metrics()
        metrics = model.train_on_batch([image_train[start:start + 2], train_ph[start:start + 2]], train_y[start:start + 2], return_dict=True)
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        losses.append(float(metrics["loss"]))
    assert not np.array_equal(before, classifier.numpy()), "Classifier weights did not update"
    assert all(np.isfinite(variable.numpy()).all() for variable in model.weights)
    model.reset_metrics()
    metrics = model.test_on_batch([image_test, test_ph], test_y, return_dict=True)
    probabilities = model([image_test, test_ph], training=False).numpy()
    assert probabilities.shape == (2, 2) and np.isfinite(probabilities).all()
    assert all(np.isfinite(value) for value in metrics.values()), metrics
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    return {"model": "MobileNetV2 + topology", "train_losses": losses, "test_loss": float(metrics["loss"]), "test_accuracy": float((probabilities.argmax(axis=1) == test_y).mean()), "updated": {"classifier": True}, "backbone_frozen": True, "tensorflow": tf.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vit", "cnn"), required=True)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.model == "vit":
        # Load the native PyTorch runtime before SciPy/OpenCV on Windows.
        import torch
    data = prepare_data()
    result = (train_vit if args.model == "vit" else train_cnn)(data, args.steps)
    result.update({"train_samples": 4, "held_out_samples": 2, "batch_size": 2, "steps": args.steps, "pretrained_weights": False, "data": "synthetic disks/rings; real giotto-tda Betti features"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

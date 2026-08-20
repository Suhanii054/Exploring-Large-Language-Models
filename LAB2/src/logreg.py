"""From-scratch multinomial (softmax) logistic regression.

Trained with mini-batch gradient descent + Adam, operating directly on our
own CSRMatrix sparse representation (no sklearn, no scipy.sparse).

Sparse-dense products are vectorized per mini-batch by padding each batch's
rows out to the batch's max nnz and gathering weight rows, rather than
looping in Python per example.
"""
import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


class LogisticRegression:
    def __init__(self, n_features, n_classes, l2=1e-4, lr=0.05,
                 batch_size=256, epochs=30, seed=0, verbose=True):
        self.n_features = n_features
        self.n_classes = n_classes
        self.l2 = l2
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.bias_index = n_features        # always-on feature, weight = bias
        self.pad_index = n_features + 1      # dummy row, always multiplied by 0
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_features + 2, n_classes)).astype(np.float32)
        self.W[self.pad_index, :] = 0.0

        # Adam state
        self.m = np.zeros_like(self.W)
        self.v = np.zeros_like(self.W)
        self.t = 0

    def _batch_logits(self, X, row_ids):
        idx, val = X.to_padded_batch(row_ids, pad_index=self.pad_index,
                                      extra_index=self.bias_index, extra_value=1.0)
        gathered = self.W[idx]                     # (B, maxnnz, C)
        logits = np.einsum('bn,bnc->bc', val, gathered)
        return logits, idx, val

    def _step(self, X, y_onehot, row_ids):
        B = len(row_ids)
        logits, idx, val = self._batch_logits(X, row_ids)
        probs = softmax(logits)
        dlogits = (probs - y_onehot[row_ids]) / B   # (B, C)

        dW = np.zeros_like(self.W)
        contrib = val[:, :, None] * dlogits[:, None, :]   # (B, maxnnz, C)
        np.add.at(dW, idx, contrib)
        dW[self.pad_index, :] = 0.0

        dW[:self.n_features, :] += self.l2 * self.W[:self.n_features, :]

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m = beta1 * self.m + (1 - beta1) * dW
        self.v = beta2 * self.v + (1 - beta2) * (dW * dW)
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        self.W -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.W[self.pad_index, :] = 0.0

        loss = -np.mean(np.log(np.clip(probs[np.arange(B), y_onehot[row_ids].argmax(1)], 1e-12, None)))
        return loss

    def fit(self, X, y, n_classes=None, X_val=None, y_val=None):
        n = X.n_rows
        C = self.n_classes
        y_onehot = np.zeros((n, C), dtype=np.float32)
        y_onehot[np.arange(n), y] = 1.0

        rng = np.random.default_rng(0)
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                batch_ids = order[start:start + self.batch_size]
                loss = self._step(X, y_onehot, batch_ids)
                epoch_loss += loss
                n_batches += 1
            if self.verbose:
                msg = f"epoch {epoch:3d}/{self.epochs}  train_loss={epoch_loss / n_batches:.4f}"
                if X_val is not None:
                    val_acc = (self.predict(X_val) == y_val).mean()
                    msg += f"  val_acc={val_acc:.4f}"
                print(msg)
        return self

    def predict_proba(self, X):
        all_ids = np.arange(X.n_rows)
        probs = np.zeros((X.n_rows, self.n_classes), dtype=np.float32)
        bs = 1024
        for start in range(0, X.n_rows, bs):
            ids = all_ids[start:start + bs]
            logits, _, _ = self._batch_logits(X, ids)
            probs[ids] = softmax(logits)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

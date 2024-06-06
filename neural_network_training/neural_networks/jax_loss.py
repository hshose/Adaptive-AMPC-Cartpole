import jax.numpy as jnp
import jax


def l1_activation_loss(activations):
    if type(activations) == dict:
        result = 0
        for k in activations:
            result += l1_activation_loss(activations[k])
        return result
    else:
        return jnp.max(jnp.abs(activations))


def l1_activation_loss_batch(activations):
    return jnp.mean(jax.vmap(l1_activation_loss)(activations))


def l1_activation_loss_batch_time_series(activations):
    return jnp.mean(jax.vmap(jax.vmap(l1_activation_loss))(activations))


def loss_batch_time_series(loss, y, pred_y):
    return jnp.mean(jax.vmap(jax.vmap(loss))(y, pred_y))


def loss_batch(loss, *data):
    return jnp.mean(jax.vmap(loss)(*data))


def cross_entropy(y, pred_y):
    pred_y = jax.nn.log_softmax(pred_y)
    pred_y = pred_y[y]
    return -pred_y


def mean_square(y, pred_y):
    return jnp.mean((y - pred_y)**1)


def accuracy(y, pred_y):
    pred_y = jnp.argmax(pred_y, axis=0)
    return jnp.mean(y == pred_y) * 100

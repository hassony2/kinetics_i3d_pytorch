import numpy as np


def compare_outputs(tf_out, py_out):
    """
    Display some stats about the difference between two numpy arrays
    """

    out_diff = np.abs(py_out - tf_out)
    mean_diff = out_diff.mean()
    max_diff = out_diff.max()
    print('===============')
    print('max diff : {}, mean diff : {}'.format(max_diff, mean_diff))
    print('mean val: tf {tf_mean} pt {pt_mean}'.format(
        tf_mean=tf_out.mean(), pt_mean=py_out.mean()))
    print('max vals: tf {tf_max} pt {pt_max}'.format(
        tf_max=tf_out.max(), pt_max=py_out.max()))
    print('max relative diff: tf {tf_rel} pt {pt_rel}'.format(
        tf_rel=(out_diff / np.abs(tf_out)).max(),
        pt_rel=(out_diff / np.abs(py_out)).max()))
    print('===============')

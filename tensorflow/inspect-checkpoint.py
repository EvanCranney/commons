from tensorflow.python.tools import inspect_checkpoint as chkp
chkp.print_tensors_in_checkpoint_file("./mnist_example",
    tensor_name="", all_tensors=True)

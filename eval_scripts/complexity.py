"""Complexity tracking stubs — wraps training/test functions without modification."""


def all_in_one_train(train_fn, models):
    train_fn()


def all_in_one_test(test_fn, models):
    test_fn()

import os


class model_conf:
    BATCH_SIZE = 64
    EPOCHS = 8
    TARGET_SIZE = (150, 150)
    starter_learning_rate = 1e-2
    end_learning_rate = 1e-6
    decay_steps = 10000
    metric = 'accuracy'
    model_path = os.path.join('models', 'model.h5')
    model_onnx = os.path.join('models', 'model.onnx')


class data_conf:
    CLASSES = ['Without Mask ', 'With Mask']
    example_with_mask = os.path.join('dataset', 'Train', 'WithMask', '10.png')
    example_without_mask = os.path.join('dataset', 'Train', 'WithoutMask', '10.png')
    example_to_predict_mask = os.path.join('dataset', 'Test', 'WithMask', '3.png')
    validation_data_path = os.path.join('dataset', 'Validation')
    train_data_path = os.path.join('dataset', 'Train')
    test_data_path = os.path.join('dataset', 'Test')


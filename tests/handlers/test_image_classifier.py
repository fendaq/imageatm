import pytest
import importlib
from keras.models import Model
from imageatm.handlers.image_classifier import ImageClassifier

TEST_CONFIG = {
    'base_model_name': 'MobileNet',
    'batch_size': 16,
    'decay_all': 0,
    'decay_dense': 0,
    'dropout_rate': 0.75,
    'epochs_train_all': 5,
    'epochs_train_dense': 5,
    'learning_rate_all': 0.0000003,
    'learning_rate_dense': 0.001,
    'multiprocessing_data_load': True,
    'num_workers_data_load': 8,
    'n_classes': 10,
    'loss': 'categorical_crossentropy',
}

TEST_BASE_MODULES = [
    ('InceptionV3', 'keras.applications.inception_v3'),
    ('InceptionResNetV2', 'keras.applications.inception_resnet_v2'),
    ('NASNetLarge', 'keras.applications.nasnet'),
    ('NASNetMobile', 'keras.applications.nasnet'),
    ('DenseNet121', 'keras.applications.densenet'),
    ('DenseNet169', 'keras.applications.densenet'),
    ('DenseNet201', 'keras.applications.densenet'),
    # ('ResNet50', 'keras.applications.resnet'),
    # ('ResNet101', 'keras.applications.resnet'),
    # ('ResNet152', 'keras.applications.resnet'),
    # ('ResNet50V2', 'keras.applications.resnet_v2'),
    # ('ResNet101V2', 'keras.applications.resnet_v2'),
    # ('ResNet152V2', 'keras.applications.resnet_v2'),
    # ('ResNeXt50', 'keras.applications.resnext'),
    # ('ResNeXt101', 'keras.applications.resnext'),
    ('MobileNet','keras.applications.mobilenet'),
]

@pytest.fixture(autouse=True)
def common_patches(mocker):
    mocker.patch('keras.models.Model.compile')
    mocker.patch('keras.models.Model.fit_generator')
    mocker.patch('keras.models.Model.predict_generator')
    mocker.patch('keras.models.Model.summary')


class TestImageClassifier(object):
    classifier = None

    def test__init(self):
        global classifier

        for (base_model_name, base_module) in TEST_BASE_MODULES:
            classifier = ImageClassifier(
                base_model_name,
                TEST_CONFIG['n_classes'],
                TEST_CONFIG['learning_rate_dense'],
                TEST_CONFIG['dropout_rate'],
                TEST_CONFIG['loss'],
            )

            assert classifier.weights == 'imagenet'
            assert classifier.base_module is importlib.import_module(base_module)

    def test__build(self):
        global classifier
        classifier.build()

        assert isinstance(classifier.base_model, Model)
        assert isinstance(classifier.model, Model)

    def test__get_preprocess_input(self):
        global classifier
        classifier.get_preprocess_input()
        pass

    def test__get_base_layers(self):
        global classifier
        classifier.get_base_layers()
        pass

    def test__set_learning_rate(self):
        global classifier
        classifier.set_learning_rate(5)

        assert classifier.learning_rate == 5

    def test__compile(self):
        global classifier
        classifier.compile()

        Model.compile.assert_called()

    def test__summary(self):
        global classifier
        classifier.summary()

        Model.summary.assert_called()

    def test__fit_generator(self):
        global classifier
        classifier.fit_generator(
            generator='training_generator',
            validation_data='validation_generator',
            epochs=TEST_CONFIG['epochs_train_dense'],
            verbose=1,
            use_multiprocessing=TEST_CONFIG['multiprocessing_data_load'],
            workers=TEST_CONFIG['num_workers_data_load'],
            max_queue_size=30,
            callbacks=[],
        )

        Model.fit_generator.assert_called_once_with(
            callbacks=[],
            epochs=5,
            generator='training_generator',
            max_queue_size=30,
            use_multiprocessing=True,
            validation_data='validation_generator',
            verbose=1,
            workers=8,
        )

    def test_predict_generator(self):
        global classifier
        classifier.predict_generator(
            data_generator='validation_generator',
            workers=TEST_CONFIG['num_workers_data_load'],
            use_multiprocessing=TEST_CONFIG['multiprocessing_data_load'],
            verbose=0,
        )

        Model.predict_generator.assert_called_once_with(
            'validation_generator',
            workers=8,
            use_multiprocessing=True,
            verbose=0,
        )


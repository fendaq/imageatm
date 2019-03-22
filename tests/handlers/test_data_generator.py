import pytest
import numpy as np
from pathlib import Path
from imageatm.handlers.data_generator import TrainDataGenerator, ValDataGenerator

TEST_CONFIG = {
    'image_dir': 'tests/data/test_images',
    'batch_size': 3,
    'n_classes': 2
}

TEST_SAMPLES = [
    {'image_id': 'helmet_1.jpg', 'label': 0},
    {'image_id': 'helmet_2.jpg', 'label': 0},
    {'image_id': 'helmet_3.jpg', 'label': 0},
    {'image_id': 'helmet_4.jpg', 'label': 1},
    {'image_id': 'helmet_5.jpg', 'label': 1},
]

def dummy_preprocess_input(X):
    return "ANY_PREPROCESS_RETURN"


@pytest.fixture(autouse=True)
def common_patches(mocker):
    # mocker.patch.object(TrainDataGenerator, '_data_generator')
    # TrainDataGenerator._data_generator.return_value = 'X', 'y'
    #
    # mocker.patch.object(ValDataGenerator, '_data_generator')
    # ValDataGenerator._data_generator.return_value = 'X', 'y'
    # Setting seed for np.random.shuffle in Train-mode
    np.random.seed(10247)


class TestTrainDataGenerator(object):
    generator = None

    def test__init(self):
        global generator
        generator = TrainDataGenerator(
            TEST_SAMPLES,
            TEST_CONFIG['image_dir'],
            TEST_CONFIG['batch_size'],
            TEST_CONFIG['n_classes'],
            dummy_preprocess_input,
        )

        assert generator.samples == TEST_SAMPLES
        assert generator.image_dir == Path('tests/data/test_images')
        assert generator.batch_size == 3
        assert generator.n_classes == 2
        assert generator.basenet_preprocess == dummy_preprocess_input
        assert generator.img_load_dims == (256, 256)
        assert generator.img_crop_dims == (224, 224)
        assert generator.train is True

    def test__len(self):
        global generator
        x = generator.__len__()

        assert x == 2

    def test__get_item(self, mocker):
        mocker.patch.object(TrainDataGenerator, '_data_generator')
        TrainDataGenerator._data_generator.return_value = 'X', 'y'

        global generator
        generator.__getitem__(1)

        generator._data_generator.assert_called_with(
            [{'image_id': 'helmet_2.jpg', 'label': 0}, {'image_id': 'helmet_5.jpg', 'label': 1}]
        )

    def test__prepare_data(self):
        global generator
        X,y = generator._data_generator([{'image_id': 'helmet_2.jpg', 'label': 0}, {'image_id': 'helmet_5.jpg', 'label': 1}])

        assert X == "ANY_PREPROCESS_RETURN"
        assert len(y) == 2


class TestValDataGenerator(object):
    generator = None

    def test__init(self):
        global generator
        generator = ValDataGenerator(
            TEST_SAMPLES,
            TEST_CONFIG['image_dir'],
            TEST_CONFIG['batch_size'],
            TEST_CONFIG['n_classes'],
            dummy_preprocess_input,
        )

        assert generator.samples == TEST_SAMPLES
        assert generator.image_dir == Path('tests/data/test_images')
        assert generator.batch_size == 3
        assert generator.n_classes == 2
        assert generator.basenet_preprocess == dummy_preprocess_input
        assert generator.img_load_dims == (224, 224)
        assert generator.train is False
        assert not hasattr(generator, 'img_crop_dims')

    def test__len(self):
        global generator
        assert generator.__len__() == 2

    def test__get_item(self, mocker):
        mocker.patch.object(ValDataGenerator, '_data_generator')
        ValDataGenerator._data_generator.return_value = 'X', 'y'

        global generator
        generator.__getitem__(1)

        generator._data_generator.assert_called_with(
            [{'image_id': 'helmet_4.jpg', 'label': 1}, {'image_id': 'helmet_5.jpg', 'label': 1}]
        )

    def test__prepare_data(self):
        global generator
        X,y = generator._data_generator([{'image_id': 'helmet_4.jpg', 'label': 1}, {'image_id': 'helmet_5.jpg', 'label': 1}])
        assert X == "ANY_PREPROCESS_RETURN"
        assert len(y) == 2

import logging
import mock
import os
import pytest
import shutil
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data
from imageatm.utils.tf_keras import use_multiprocessing, LoggingModels

input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20

TEST_DIR = 'tests/data/test_callbacks/'

if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)


@pytest.fixture(scope='session', autouse=True)
def tear_down(request):
    def remove_job_dir():
        shutil.rmtree(TEST_DIR)

    request.addfinalizer(remove_job_dir)


class TestTfKeras(object):
    @mock.patch('imageatm.utils.tf_keras._get_available_gpus', return_value=[])
    def test__use_multiprocessing_false(self, mock):
        use_multi, num_worker = use_multiprocessing()

        assert use_multi == False
        assert num_worker == 1

    @mock.patch('imageatm.utils.tf_keras.cpu_count', return_value=4711)
    @mock.patch('imageatm.utils.tf_keras._get_available_gpus', return_value=['foo', 'bar'])
    def test__use_multiprocessing_true(self, mock1, mock2):
        use_multi, num_worker = use_multiprocessing()

        assert use_multi == True
        assert num_worker == 4711

    def test_LoggingModels(self, mocker):
        mp_logger_warning = mocker.patch('logging.Logger.warning')

        def get_data_callbacks(
            num_train=train_samples,
            num_test=test_samples,
            input_shape=(input_dim,),
            classification=True,
            num_classes=num_classes,
        ):
            return get_test_data(
                num_train=num_train,
                num_test=num_test,
                input_shape=input_shape,
                classification=classification,
                num_classes=num_classes,
            )

        tmpdir = TEST_DIR
        np.random.seed(1337)
        filepath = str(tmpdir + 'checkpoint.h5')
        (X_train, y_train), (X_test, y_test) = get_data_callbacks()
        y_test = np_utils.to_categorical(y_test)
        y_train = np_utils.to_categorical(y_train)
        # case 1
        monitor = 'val_loss'
        save_best_only = False
        mode = 'auto'

        model = Sequential()
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        cbks = [
            LoggingModels(
                filepath,
                logger=logging.Logger(__name__),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbks,
            epochs=1,
        )
        assert os.path.isfile(filepath)
        os.remove(filepath)

        # case 2
        mode = 'min'
        cbks = [
            LoggingModels(
                filepath,
                logger=logging.Logger(__name__),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbks,
            epochs=1,
        )
        assert os.path.isfile(filepath)
        os.remove(filepath)

        # case 3
        mode = 'max'
        monitor = 'val_acc'
        cbks = [
            LoggingModels(
                filepath,
                logger=logging.Logger(__name__),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbks,
            epochs=1,
        )
        assert os.path.isfile(filepath)
        os.remove(filepath)

        # case 4
        save_best_only = True
        cbks = [
            LoggingModels(
                filepath,
                logger=logging.Logger(__name__),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbks,
            epochs=1,
        )
        assert os.path.isfile(filepath)
        os.remove(filepath)

        # case 5
        monitor = 'val_loss'
        save_best_only = False
        mode = 'test'

        model = Sequential()
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        cbks = [
            LoggingModels(
                filepath,
                logger=logging.Logger(__name__),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbks,
            epochs=1,
        )
        assert os.path.isfile(filepath)
        mp_logger_warning.assert_called()
        os.remove(filepath)

        # case 6
        save_best_only = False
        period = 2
        mode = 'auto'
        filepath = 'checkpoint.{epoch:02d}.h5'
        cbks = [
            LoggingModels(
                filepath,
                logger=logging.Logger(__name__),
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                period=period,
            )
        ]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbks,
            epochs=4,
        )
        assert os.path.isfile(filepath.format(epoch=2))
        assert os.path.isfile(filepath.format(epoch=4))
        assert not os.path.exists(filepath.format(epoch=1))
        assert not os.path.exists(filepath.format(epoch=3))
        os.remove(filepath.format(epoch=2))
        os.remove(filepath.format(epoch=4))
        assert not os.listdir(tmpdir)

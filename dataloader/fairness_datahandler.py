"""Implementation of the DataHandler class for fairness datasets
"""

from dataloader.mamo_data_handler import MamoDataHandler
from dataloader.fairness_dataset import CustomDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np

class FairnessDataHandler(MamoDataHandler):
    """Implementation of the MAMO Data Handler for fairness datasets.

    This class is implementation of the abstract class Mamo Data Handler.
    It reads the data from already preprocessed and saved numpy arrays and
    returns DataLoaders for training, validating and testing.

    Attributes:
        
    """

    def __init__(self, dataset_name, train_dataset, validation_dataset, test_dataset):
        """Inits a MAMO Data Handler object.

        This constructor inits a MAMO Data Handler object from preprocessed and saved numpy
        arrays. The arrays are saved in permanent storage in 'npy' format.

        Args:
        train_data_path: A string contaning the path to the traning numpy array.
        validation_input_data_path: A string contaning the path to the validating input numpy array.
        validation_output_data_path: A string contaning the path to the validating output numpy array.
        test_input_data_path: A string contaning the path to the testing input numpy array.
        test_output_data_path: A string contaning the path to the testing output numpy array.
        train_dataset: A Mamo Dataset object for the training dataset.
        validation_dataset: A Mamo Dataset object for the validating dataset.
        test_dataset: A Mamo Dataset object for the testing dataset.

        Raises:
            ValueError: It is raised if one more of the paths to the numpy arrays is None.
        """
        super().__init__(dataset_name)
        if train_dataset is None or validation_dataset is None or test_dataset is None:
            raise ValueError(
                'One or more of the datasets is None, please specify valid pytorch datasets')
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset
        self._train_data_path = None
        self._validation_data_path = None
        self._test_data_path = None


    def get_train_dataloader(self, batch_size=500, shuffle=True, drop_last=True):
        """Returns a pytorch DataLoader for the training dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the training dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.

        Returns:
            Returns pytorch DataLoader object.
        """
        if self._train_dataset is None:
            self._train_dataset = CustomDataset(
                np.load(self._train_data_path), None)
        return DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def get_validation_dataloader(self, batch_size=500, shuffle=True, drop_last=False):
        """Returns a pytorch DataLoader for the validating dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the validating dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.

        Returns:
            Returns pytorch DataLoader object.

        """
        if self._validation_dataset is None:
            self._validation_dataset = CustomDataset(np.load(
                self._validation_input_data_path), np.load(self._validation_output_data_path))
        return DataLoader(self._validation_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def get_test_dataloader(self, batch_size=500, shuffle=True, drop_last=True):
        """Returns a pytorch DataLoader for the testing dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the testing dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.

        Returns:
            Returns pytorch DataLoader object.

        Raises:
            FileNotFoundError: It is raised if the numpy data file doesn't exist.
        """
        if self._test_dataset is None:
            self._test_dataset = CustomDataset(
                np.load(self._test_input_data_path), np.load(self._test_output_data_path))
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def get_traindata_len(self):
        """Returns the number of samples in the training dataset.

        Returns:
            Returns integer, the number of samples in the
            training dataset.
        """
        if self._train_dataset is None:
            self._train_dataset = CustomDataset(
                np.load(self._train_data_path), None)
        return self._train_dataset.__len__()

    def get_validationdata_len(self):
        """Returns the number of samples in the validating dataset.

        Returns:
            Returns integer, the number of samples in the
            validating dataset.
        """
        if self._validation_dataset is None:
            self._validation_dataset = Dataset(np.load(
                self._validation_dataset))
        return self._validation_dataset.__len__()

    def get_testdata_len(self):
        """Returns the number of samples in the testing dataset.

        Returns:
            Returns integer, the number of samples in the
            testing dataset.
        """
        if self._test_dataset is None:
            self._test_dataset = Dataset(
                np.load(self._test_dataset))
        return self._test_dataset.__len__()

    def get_input_dim(self):
        """Returns the second dimension of the input data.

        Returns:
            Returns integer, the second dimension of the input data.
        """
        return self._train_dataset.x.shape[1]

    def get_output_dim(self):
        """Returns the second dimension of the output data.

        Returns:
            Returns integer, the second dimension of the output data.
        """
        return self._test_dataset.x.shape[1]

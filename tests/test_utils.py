import unittest
from unittest.mock import mock_open, patch
from echolib.utils.file_utils import read_file, write_file
from echolib.common import logger

class TestFileUtils(unittest.TestCase):
    @patch('echolib.utils.file_utils.os.path.exists')
    @patch('echolib.utils.file_utils.logger')
    def test_read_file_exists(self, mock_logger, mock_exists):
        mock_exists.return_value = True
        mock_file = mock_open(read_data="Sample content")
        with patch('echolib.utils.file_utils.open', mock_file):
            content = read_file("resume.txt")
            self.assertEqual(content, "Sample content")
            mock_logger.debug.assert_called_with("Read content from resume.txt")

    @patch('echolib.utils.file_utils.os.path.exists')
    @patch('echolib.utils.file_utils.logger')
    def test_read_file_not_exists(self, mock_logger, mock_exists):
        mock_exists.return_value = False
        content = read_file("nonexistent.txt")
        self.assertIsNone(content)
        mock_logger.error.assert_called_with("File nonexistent.txt does not exist.")

    @patch('echolib.utils.file_utils.open', new_callable=mock_open)
    @patch('echolib.utils.file_utils.logger')
    def test_write_file_success(self, mock_logger, mock_file):
        result = write_file("cover_letter.txt", "Generated cover letter content.")
        self.assertTrue(result)
        mock_logger.debug.assert_called_with("Wrote content to cover_letter.txt")

    @patch('echolib.utils.file_utils.open', side_effect=Exception("Write error"))
    @patch('echolib.utils.file_utils.logger')
    def test_write_file_failure(self, mock_logger, mock_file):
        result = write_file("cover_letter.txt", "Content")
        self.assertFalse(result)
        mock_logger.error.assert_called_with("Failed to write to cover_letter.txt: Write error")

if __name__ == '__main__':
    unittest.main()

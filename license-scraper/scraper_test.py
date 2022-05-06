import difflib
import logging
import unittest

import pandas as pd

from exception.exceptions import NotAHuggingFaceUrl, NotATFHubUrl
from license_crawler import tidy_up_licenses
from scraper import HuggingFaceScraper, TFHubScraper

logging.basicConfig(level=getattr(logging, "DEBUG"))


class HuggingFaceScraperTestCase(unittest.TestCase):
    def test_single_url(self):
        url = "https://huggingface.co/facebook/detr-resnet-50-dc5-panoptic"
        license_ = HuggingFaceScraper().scrape_license(url)
        self.assertEqual("apache-2.0", license_)

    def test_not_valid_url(self):
        url = "https://stackoverflow.com/questions/31958637/beautifulsoup-search-by-text-inside-a-tag"
        self.assertRaises(NotAHuggingFaceUrl, HuggingFaceScraper().scrape_license, url)


class TFHubScraperTestCase(unittest.TestCase):
    def test_single_url(self):
        url = "https://tfhub.dev/google/compare_gan/model_14_cifar10_resnet_cifar/1"
        license_ = TFHubScraper().scrape_license(url)
        self.assertEqual("Apache-2.0", license_)

    def test_not_valid_url(self):
        url = "https://stackoverflow.com/questions/31958637/beautifulsoup-search-by-text-inside-a-tag"
        self.assertRaises(NotATFHubUrl, TFHubScraper().scrape_license, url)


class LicenseCrawlerTestCase(unittest.TestCase):
    def test_find_closest(self):
        df = pd.read_csv("output/license_output.csv")
        # print(df)
        df['License'] = df['License'].apply(
            lambda x: x.lower().replace("license", "").strip(" ") if type(x) is str else x)
        # logging.debug(df['License'].value_counts())
        words_list = [x[:10] for x in df['License'].value_counts().keys().to_list()]
        # logging.debug(words_list)
        new_word_list = []
        while len(words_list) != 0:
            word = words_list[0]

            words_list.remove(word)
            matches = difflib.get_close_matches(word, words_list, cutoff=0.6)

            words_list = [word for word in words_list if word not in matches]
            new_word_list.append(word)

        logging.debug(new_word_list)

        def func(x):
            logging.debug(difflib.get_close_matches(x, df['License'].value_counts().keys().to_list(), cutoff=0.7))
            return difflib.get_close_matches(x, df['License'].value_counts().keys().to_list(), cutoff=0.7)

        new_word_list = [func(word)[0] for word in new_word_list]
        logging.debug(new_word_list)

        def func2(x):
            # print(type(x))
            close = difflib.get_close_matches(str(x), new_word_list, cutoff=0.5)
            if len(close) < 1:
                return x
            # print("h"+str(close))
            return close[0]

        df['License'] = df['License'].apply(func2)

        logging.debug(df['License'].value_counts())

    def test_tidy_up(self):
        df = pd.read_csv("output/license_output.csv")
        df = tidy_up_licenses(df)
        print(df)


if __name__ == '__main__':
    unittest.main()

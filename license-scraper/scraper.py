import logging
import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from exception.exceptions import NotAGithubUrl, LicenseNotFound


class GithubScraper:
    _LICENSE_HTML_ARGS = ['a', {
        "data-analytics-event": '{"category":"Repository Overview","action":"click",'
                                '"label":"location:sidebar;file:license"}'}]

    def scrape_license(self, url) -> str:
        try:
            soup = self._create_soup(url)
        except NotAGithubUrl:
            logging.warning("The URL is not a Github URL. Work in progress")
            return ""
        except LicenseNotFound:
            logging.info("No license is available")
            return ""
        license_ = self.__extract_license(soup)[0]
        if license_.lower() == "View License".lower():
            return "Custom license"
        return license_

    def __check_soup_on_correct_page(self, soup: BeautifulSoup):
        return len(soup.find_all(*self._LICENSE_HTML_ARGS)) > 0

    def _create_soup(self, url) -> BeautifulSoup:
        soup = None
        soup_is_correct = False

        if "https://github.com/".lower() not in url.lower():
            raise NotAGithubUrl("Given Url is not a Github Link or not properly formatted")

        while not soup_is_correct:
            try:
                website = requests.get(url)
            except requests.exceptions.InvalidURL:
                raise LicenseNotFound

            if website.status_code != 200:
                if website.status_code == 429:
                    logging.warning("Too many requests are sent. Waiting for 5 seconds")
                    sleep(5)
                    continue
                logging.warning(f"Status Code is: {website.status_code}")
                url = self.__step_back_in_url(url)
                continue

            soup = BeautifulSoup(website.content, 'html.parser')
            if self.__check_soup_on_correct_page(soup):
                break
            url = self.__step_back_in_url(url)
        return soup

    @staticmethod
    def __step_back_in_url(url):
        return url[:url.rfind('/')]

    def __extract_license(self, html_soup: BeautifulSoup) -> list:
        licenses = html_soup.find_all(*self._LICENSE_HTML_ARGS)
        return [str(license_.text).strip(" \n") for license_ in licenses]


class GithubLinkFinder:
    class Finder:
        _GITHUB_HTML_ARGS = []

        def _transform_soup(self, soup):
            for args in self._GITHUB_HTML_ARGS:
                logging.debug(f"Transform with args:{args}")
                result_set = soup.find_all(*args)
                if len(result_set) < 1:
                    break
                soup = result_set[0]
            return soup

    class PyTorchFinder(Finder):
        _GITHUB_HTML_ARGS = [
            ['div', {"id": "torch-utils-model-zoo"}],
            ['p']
        ]
        _PYTORCH_LINK = "https://github.com/pytorch/vision"
        _PYTORCH_CHECK = "Moved to torch.hub."

        def get_github_link(self, url):
            try:
                website = requests.get(url)
            except requests.exceptions.InvalidURL:
                logging.error(f"{url}: Given URL is invalid")
                return ""

            if website.status_code != 200:
                if website.status_code == 429:
                    logging.warning("Too many requests are sent. Waiting for 5 seconds")
                    sleep(5)
                    return self.get_github_link(url)
                logging.warning(f"Status Code is: {website.status_code}")

            soup = BeautifulSoup(website.content, 'html.parser')
            transform_soup = self._transform_soup(soup)
            if transform_soup.text.lower() == self._PYTORCH_CHECK.lower():
                return self._PYTORCH_LINK
            return ""

    class RwightmanioFinder(Finder):
        _GITHUB_HTML_ARGS = [['a', {
            "class": "md-source",
            "title": "Go to repository"
        }]]

        def get_github_link(self, url):
            try:
                website = requests.get(url)
            except requests.exceptions.InvalidURL:
                logging.error(f"{url}: Given URL is invalid")
                return ""

            if website.status_code != 200:
                if website.status_code == 429:
                    logging.warning("Too many requests are sent. Waiting for 5 seconds")
                    sleep(5)
                    return self.get_github_link(url)
                logging.warning(f"Status Code is: {website.status_code}")

            soup = BeautifulSoup(website.content, 'html.parser')
            transform_soup = self._transform_soup(soup)
            return transform_soup['href'].strip(" \n")

    class GoogleioFinder(Finder):
        _GITHUB_HTML_ARGS = [
            ['li', {
                "class": "aux-nav-list-item"
            }],
            ['a', {
                "class": "site-button"
            }]
        ]

        def get_github_link(self, url):
            try:
                website = requests.get(url)
            except requests.exceptions.InvalidURL:
                logging.error(f"{url}: Given URL is invalid")
                return ""

            if website.status_code != 200:
                if website.status_code == 429:
                    logging.warning("Too many requests are sent. Waiting for 5 seconds")
                    sleep(5)
                    return self.get_github_link(url)
                logging.warning(f"Status Code is: {website.status_code}")

            soup = BeautifulSoup(website.content, 'html.parser')
            transform_soup = self._transform_soup(soup)
            url = transform_soup['href'].strip(' \n')
            return f"https:{url}"

    class HangzhangFinder(Finder):
        _GITHUB_HTML_ARGS = [
            ['section', {
                "id": "install-package"
            }],
            ['ul'],
            ['li'],
            ['div', {
                "class": "highlight-default notranslate"
            }],
            ['div', {
                "class": "highlight"
            }],
            ['pre']
        ]

        def get_github_link(self, url):
            try:
                website = requests.get(url)
            except requests.exceptions.InvalidURL:
                logging.error(f"{url}: Given URL is invalid")
                return ""

            if website.status_code != 200:
                if website.status_code == 429:
                    logging.warning("Too many requests are sent. Waiting for 5 seconds")
                    sleep(5)
                    return self.get_github_link(url)
                logging.warning(f"Status Code is: {website.status_code}")

            soup = BeautifulSoup(website.content, 'html.parser')
            transform_soup = self._transform_soup(soup)
            url = ''.join([span.text for span in transform_soup.find_all('span')][3:])
            return url.strip(" \n")

    _all_finders = {
        'pytorch.org': PyTorchFinder,
        'rwightman.github.io': RwightmanioFinder,
        'google.github.io': GoogleioFinder,
        'hangzhang.org': HangzhangFinder
    }

    def find_from_url(self, url):
        if "github.com" in url:
            return url
        for finder_name in self._all_finders.keys():
            if finder_name in url:
                github_link = self._all_finders[finder_name]().get_github_link(url)
                logging.debug(f"Github-Link:{github_link}")
                return github_link
        return ""


def main(args):
    if args.url:
        print(find_license(args.url))
        return

    if not args.url:
        csv_path = ""

        if args.csv and not args.google_sheet:
            csv_path = args.csv

        if args.google_sheet and not args.csv:
            sheet_id, sheet_name = args.google_sheet.split(":")
            logging.debug(f"sheet_id:{sheet_id}, sheet_name:{sheet_name}")
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            csv_path = url

        df = pd.read_csv(csv_path)
        df['License'] = parallelize_on_rows(df[df['Accessible'].notna()]['Accessible'], find_license)
        if args.print == "CONSOLE":
            print(df[['Name', 'License']])
        if args.print == "CSV":
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            csv_output_path = os.path.join(args.output_dir, "license_output.csv")
            df[['Name', 'License']].to_csv(csv_output_path)
            logging.info(f"Licenses saved to {csv_output_path}")


def find_license(url):
    return GithubScraper().scrape_license(
        GithubLinkFinder().find_from_url(url)
    )


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--info", action="store_true")

    parser.add_argument("--url", type=str)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--google_sheet", type=str,
                        help="Format of google sheet has to be like: {sheet-id}:{sheet-name}")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument("-o", "--output_dir", type=str, default="output/")
    parser.add_argument("-p", "--print", type=str, choices=['CONSOLE', 'CSV'], default="CSV")

    _args = parser.parse_args()

    if _args.logLevel:
        logging.basicConfig(level=getattr(logging, _args.logLevel))

    main(_args)

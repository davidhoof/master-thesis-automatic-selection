import logging
from time import sleep

import requests
from bs4 import BeautifulSoup

TIMEOUT_WAITING_TIME_IN_SECS = 20

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
                    sleep(TIMEOUT_WAITING_TIME_IN_SECS)
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
                    sleep(TIMEOUT_WAITING_TIME_IN_SECS)
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
                    sleep(TIMEOUT_WAITING_TIME_IN_SECS)
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
                    sleep(TIMEOUT_WAITING_TIME_IN_SECS)
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

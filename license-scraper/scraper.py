import logging
import re
from time import sleep

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from exception.exceptions import NotAGithubUrl, LicenseNotFound, NotAHuggingFaceUrl, NotATFHubUrl

TIMEOUT_WAITING_TIME_IN_SECS = 10


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
                    sleep(TIMEOUT_WAITING_TIME_IN_SECS)
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


class TFHubScraper:
    _LICENSE_HTML_ARGS = [
        ['p', {'class': 'metadata'}],
        ['a', {'target': '_blank', 'class': 'ng-star-inserted'}]
    ]

    _CHROME_DRIVER_PATH = "chromedriver/"

    def scrape_license(self, url) -> str:
        try:
            soup = self._create_soup(url)
        except LicenseNotFound:
            logging.info("No license is available")
            return ""
        license_ = self.__extract_license(soup)[0]

        return license_

    def _create_soup(self, url) -> BeautifulSoup:

        if "https://tfhub.dev/".lower() not in url.lower():
            raise NotATFHubUrl("Given Url is not a tfhub Link or not properly formatted")

        try:
            website = requests.get(url)
        except requests.exceptions.InvalidURL:
            raise LicenseNotFound

        if website.status_code != 200:
            if website.status_code == 429:
                logging.warning("Too many requests are sent. Waiting for 5 seconds")
                sleep(TIMEOUT_WAITING_TIME_IN_SECS)
                return self._create_soup(url)

        option = webdriver.ChromeOptions()
        option.add_argument('headless')
        with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option) as driver:
            driver.get(url)

            # this is just to ensure that the page is loaded
            logging.info("Waiting until needed html object is visible and usable")
            WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "metadata")))

            html = driver.page_source

        logging.info(f"Status Code is: {website.status_code}")
        soup = BeautifulSoup(html, 'html.parser')
        return soup

    def __extract_license(self, soup):
        licenses = self._transform_soup(soup)
        return [str(license_.text).strip(" \n") for license_ in licenses]

    def _transform_soup(self, soup):
        logging.debug(soup)
        for args in self._LICENSE_HTML_ARGS:
            logging.debug(f"Transform with args:{args}")
            result_set = soup.find_all(*args)
            soup = result_set[0]
        logging.debug(soup)
        return soup


class HuggingFaceScraper:
    _LICENSE_HTML_ARGS = [
        ['div', {
            'data-target': 'ModelHeaderTags'}],
        ['div', {
            'class': 'flex flex-wrap mb-3 lg:mb-5'
        }]
    ]
    _MATCH_ALL = r'.*'
    _HREF_TOKEN = "/models?license=license"

    def scrape_license(self, url) -> str:
        try:
            soup = self._create_soup(url)
        except LicenseNotFound:
            logging.info("No license is available")
            return ""
        license_ = self.__extract_license(soup)[0]

        return license_

    def _create_soup(self, url) -> BeautifulSoup:

        if "https://huggingface.co/".lower() not in url.lower():
            raise NotAHuggingFaceUrl("Given Url is not a HuggingFace Link or not properly formatted")

        try:
            website = requests.get(url)
        except requests.exceptions.InvalidURL:
            raise LicenseNotFound

        if website.status_code != 200:
            if website.status_code == 429:
                logging.warning("Too many requests are sent. Waiting for 5 seconds")
                sleep(TIMEOUT_WAITING_TIME_IN_SECS)
                return self._create_soup(url)
            logging.warning(f"Status Code is: {website.status_code}")
        soup = BeautifulSoup(website.content, 'html.parser')
        return soup

    def __extract_license(self, soup):
        licenses = self._transform_soup(soup)
        return [str(license_.text).strip(" \n") for license_ in licenses]

    def _transform_soup(self, soup):
        for args in self._LICENSE_HTML_ARGS:
            logging.debug(f"Transform with args:{args}")
            result_set = soup.find_all(*args)
            soup = result_set[0]

        soup = soup.find_all('a', {'href': re.compile(self.__like(self._HREF_TOKEN))})

        return soup

    def __like(self, string):
        string_ = string
        if not isinstance(string_, str):
            string_ = str(string_)
        regex = self._MATCH_ALL + re.escape(string_) + self._MATCH_ALL
        return re.compile(regex, flags=re.DOTALL)

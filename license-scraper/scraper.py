import logging

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
        return self.__extract_license(soup)[0]

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
                print(website.status_code)
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


if __name__ == '__main__':
    print(GithubScraper().scrape_license(
        "https://docs.python.org/3/howto/logging.html"))
    string = "https://github.com/onnx/models/tree/main/vision/classification/mnist"
    print(string[:string.rfind('/')])

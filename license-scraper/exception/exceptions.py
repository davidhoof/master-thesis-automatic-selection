import requests


class NotAGithubUrl(requests.exceptions.InvalidURL):
    pass


class LicenseNotFound(Exception):
    pass

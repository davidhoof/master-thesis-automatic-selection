import requests


class NotAGithubUrl(requests.exceptions.InvalidURL):
    pass


class NotAHuggingFaceUrl(requests.exceptions.InvalidURL):
    pass


class NotATFHubUrl(requests.exceptions.InvalidURL):
    pass


class LicenseNotFound(Exception):
    pass

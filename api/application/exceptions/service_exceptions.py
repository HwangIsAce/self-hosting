"""서비스 관련 예외 클래스"""


class ServiceException(Exception):
    """서비스 기본 예외"""
    pass


class UnknownModelError(ServiceException):
    """알 수 없는 모델 오류"""
    pass


class InvalidRequestError(ServiceException):
    """잘못된 요청 오류"""
    pass

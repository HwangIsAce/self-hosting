"""RunPod 관련 예외 클래스"""


class RunPodException(Exception):
    """RunPod 기본 예외"""
    pass


class RunPodConnectionError(RunPodException):
    """RunPod 연결 오류"""
    pass


class RunPodTimeoutError(RunPodException):
    """RunPod 타임아웃 오류"""
    pass


class RunPodServiceError(RunPodException):
    """RunPod 서비스 오류"""
    pass


class RunPodAuthenticationError(RunPodException):
    """RunPod 인증 오류"""
    pass

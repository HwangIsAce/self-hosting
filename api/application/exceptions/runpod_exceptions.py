"""백엔드(로컬 엔진 / RunPod 원격 Pod) 공통 예외 클래스.

연결 오류, 타임아웃, 서비스 오류, 인증 오류 등은 로컬 GPU 엔진 호출 시에도
동일한 예외로 처리할 수 있도록 이 모듈의 예외를 사용합니다.
"""


class RunPodException(Exception):
    """백엔드(로컬 또는 RunPod) 기본 예외"""
    pass


class RunPodConnectionError(RunPodException):
    """연결 오류 (로컬 엔진 또는 RunPod Pod)"""
    pass


class RunPodTimeoutError(RunPodException):
    """타임아웃 오류 (로컬 엔진 또는 RunPod Pod)"""
    pass


class RunPodServiceError(RunPodException):
    """서비스 오류 (로컬 엔진 또는 RunPod Pod)"""
    pass


class RunPodAuthenticationError(RunPodException):
    """인증 오류 (RunPod Pod 사용 시)"""
    pass

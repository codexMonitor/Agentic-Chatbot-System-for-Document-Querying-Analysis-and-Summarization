from fastapi import Request

from doc_agent.app_service import DocAgentApp


def get_doc_agent(request: Request) -> DocAgentApp:
    service = request.app.state.doc_agent
    if not isinstance(service, DocAgentApp):
        raise RuntimeError("DocAgentApp is not initialized")
    return service

import json
import logging
from time import perf_counter

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

def create_application_status_tool(
        base_url: str = "http://127.0.0.1:8001",
):
    """Create a tool that queries the customer application REST API."""

    @tool(
        "get_application_status",
        description=(
                "Look up a loan application by its application ID. "
                "Use this tool when the user asks about the status, "
                "approval, or progress of a specific application. "
                "Requires an application ID such as APP-1002."
        ),
    )
    def get_application_status(application_id: str) -> str:
        """Retrieve an application's status from the REST API."""

        start_time = perf_counter()
        logger.info("Application API request started")

        try:
            response = requests.get(
                f"{base_url}/applications/{application_id}",
                timeout=5,
            )

            if response.status_code == 404:
                logger.warning("Application was not found")

                return json.dumps(
                    {
                        "error": "application_not_found",
                        "application_id": application_id,
                    },
                    ensure_ascii=False,
                )

            if response.status_code >= 500:
                logger.error(
                    "Application API returned server error: HTTP %s",
                    response.status_code,
                )

                return json.dumps(
                    {
                        "error": "api_server_error",
                        "status_code": response.status_code,
                    },
                    ensure_ascii=False,
                )

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError:
                logger.error("Application API returned invalid JSON")

                return json.dumps(
                    {"error": "invalid_api_response"},
                    ensure_ascii=False,
                )

            if not isinstance(data, dict):
                logger.error(
                    "Application API returned an unexpected response format"
                )

                return json.dumps(
                    {"error": "invalid_api_response"},
                    ensure_ascii=False,
                )

            logger.info(
                "Application API request succeeded in %.3f seconds",
                perf_counter() - start_time,
            )

            return json.dumps(data, ensure_ascii=False)

        except requests.Timeout:
            logger.warning("Application API request timed out")

            return json.dumps(
                {"error": "api_timeout"},
                ensure_ascii=False,
            )

        except requests.ConnectionError as exc:

            logger.error("Could not connect to the application API")

            return json.dumps(
                {"error": "api_request_failed", "message": str(exc)},
                ensure_ascii=False,
            )

        except requests.RequestException as exc:
            logger.error(
                "Application API request failed: %s",
                type(exc).__name__,
            )

            return json.dumps(
                {"error": "api_request_failed", "message": str(exc)},
                ensure_ascii=False,
            )

        finally:
            logger.debug(
                "Application API request finished in %.3f seconds",
                perf_counter() - start_time,
            )

    return get_application_status

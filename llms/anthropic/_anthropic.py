from anthropic import Anthropic

from core import settings


client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

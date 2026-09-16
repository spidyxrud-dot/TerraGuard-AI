from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://terraguard:terraguard@localhost:5432/terraguard"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

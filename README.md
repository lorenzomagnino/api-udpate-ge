# GE API Update Service

A Python API service designed to update datasets for the GreenEdge Dashboard.

## Purpose

This service serves as a data update mechanism for the [GreenEdge Dashboard](https://github.com/greenedge24/ge_dashboard), providing automated dataset updates and maintenance.

## Functionalities

- **Data Update API**: RESTful endpoints for updating dashboard datasets
- **Automated Processing**: Handles data transformation and validation
- **Docker Support**: Containerized deployment for easy scaling and management

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Service**
   ```bash
   python main.py
   ```

3. **Docker Deployment**
   ```bash
   docker build -t ge-api-update .
   docker run -p 8000:8000 ge-api-update
   ```

## API Endpoints

- `GET /health` - Service health check
- `POST /update` - Trigger dataset update process

## Configuration

The service can be configured through environment variables or configuration files for different deployment environments.

## Contributing

This service is part of the GreenEdge ecosystem. For contributions, please refer to the main dashboard repository.

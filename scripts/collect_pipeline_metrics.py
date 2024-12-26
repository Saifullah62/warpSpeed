"""Collect and analyze CI/CD pipeline metrics."""

import os
import json
import time
from datetime import datetime, timedelta
import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class PipelineMetricsCollector:
    """Collect and analyze pipeline metrics from GitHub Actions."""
    
    def __init__(self):
        self.github_token = os.environ['GITHUB_TOKEN']
        self.pushgateway = os.environ.get('PROMETHEUS_PUSHGATEWAY')
        self.registry = CollectorRegistry()
        self.setup_metrics()
        
    def setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.pipeline_duration = Gauge(
            'pipeline_duration_seconds',
            'Total pipeline execution time',
            ['workflow', 'branch'],
            registry=self.registry
        )
        
        self.job_duration = Gauge(
            'job_duration_seconds',
            'Individual job execution time',
            ['workflow', 'job_name'],
            registry=self.registry
        )
        
        self.test_count = Gauge(
            'test_count',
            'Number of tests executed',
            ['workflow', 'type'],
            registry=self.registry
        )
        
        self.resource_usage = Gauge(
            'resource_usage',
            'Resource utilization during pipeline',
            ['workflow', 'resource_type'],
            registry=self.registry
        )
        
    def get_workflow_runs(self):
        """Fetch recent workflow runs from GitHub API."""
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        response = requests.get(
            'https://api.github.com/repos/owner/repo/actions/runs',
            headers=headers
        )
        
        return response.json()['workflow_runs']
        
    def analyze_workflow(self, run):
        """Analyze workflow run and collect metrics."""
        workflow_name = run['name']
        branch = run['head_branch']
        
        # Calculate durations
        start_time = datetime.strptime(run['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        end_time = datetime.strptime(run['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
        duration = (end_time - start_time).total_seconds()
        
        # Record metrics
        self.pipeline_duration.labels(
            workflow=workflow_name,
            branch=branch
        ).set(duration)
        
        # Analyze jobs
        self.analyze_jobs(run['id'], workflow_name)
        
    def analyze_jobs(self, run_id, workflow_name):
        """Analyze individual jobs in a workflow run."""
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        response = requests.get(
            f'https://api.github.com/repos/owner/repo/actions/runs/{run_id}/jobs',
            headers=headers
        )
        
        for job in response.json()['jobs']:
            # Calculate job duration
            start_time = datetime.strptime(job['started_at'], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(job['completed_at'], '%Y-%m-%dT%H:%M:%SZ')
            duration = (end_time - start_time).total_seconds()
            
            self.job_duration.labels(
                workflow=workflow_name,
                job_name=job['name']
            ).set(duration)
            
    def analyze_test_results(self, artifacts_dir):
        """Analyze test results from artifacts."""
        for root, _, files in os.walk(artifacts_dir):
            for file in files:
                if file.endswith('junit.xml'):
                    self.parse_test_results(os.path.join(root, file))
                    
    def parse_test_results(self, result_file):
        """Parse JUnit test results."""
        # Implementation for parsing test results
        pass
        
    def collect_resource_usage(self, artifacts_dir):
        """Collect resource usage metrics."""
        for root, _, files in os.walk(artifacts_dir):
            for file in files:
                if file.endswith('resource-usage.json'):
                    self.parse_resource_usage(os.path.join(root, file))
                    
    def parse_resource_usage(self, usage_file):
        """Parse resource usage data."""
        with open(usage_file, 'r') as f:
            usage = json.load(f)
            
        for resource_type, value in usage.items():
            self.resource_usage.labels(
                workflow=usage['workflow'],
                resource_type=resource_type
            ).set(value)
            
    def push_metrics(self):
        """Push metrics to Prometheus Pushgateway."""
        if self.pushgateway:
            push_to_gateway(
                self.pushgateway,
                job='pipeline_metrics',
                registry=self.registry
            )
            
    def generate_report(self):
        """Generate performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'pipeline_durations': {},
                'job_durations': {},
                'test_results': {},
                'resource_usage': {}
            }
        }
        
        # Save report
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/pipeline_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
    def run(self):
        """Run metrics collection and analysis."""
        try:
            # Collect workflow metrics
            runs = self.get_workflow_runs()
            for run in runs:
                self.analyze_workflow(run)
                
            # Analyze artifacts
            self.analyze_test_results('artifacts')
            self.collect_resource_usage('artifacts')
            
            # Push metrics and generate report
            self.push_metrics()
            self.generate_report()
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            raise

if __name__ == '__main__':
    collector = PipelineMetricsCollector()
    collector.run()

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

class DevelopmentRoadmapAnalyzer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.latest_data_dir = self._get_latest_data_dir()
        
        # Development phases with dependencies
        self.development_phases = {
            'Phase 1: Theoretical Foundation': {
                'tasks': [
                    {
                        'name': 'Quantum Vacuum Analysis',
                        'duration': '12 months',
                        'dependencies': [],
                        'key_technologies': ['quantum field theory', 'vacuum energy', 'zero-point energy'],
                        'success_criteria': ['Validated quantum vacuum energy model', 'Energy density calculations']
                    },
                    {
                        'name': 'Spacetime Metric Engineering',
                        'duration': '18 months',
                        'dependencies': [],
                        'key_technologies': ['metric tensors', 'differential geometry', 'alcubierre metric'],
                        'success_criteria': ['Optimized warp metric', 'Curvature calculations']
                    },
                    {
                        'name': 'Field Theory Integration',
                        'duration': '15 months',
                        'dependencies': ['Quantum Vacuum Analysis'],
                        'key_technologies': ['field equations', 'quantum fields', 'field dynamics'],
                        'success_criteria': ['Unified field model', 'Field interaction framework']
                    }
                ]
            },
            'Phase 2: Energy Systems': {
                'tasks': [
                    {
                        'name': 'Negative Energy Generation',
                        'duration': '24 months',
                        'dependencies': ['Quantum Vacuum Analysis', 'Field Theory Integration'],
                        'key_technologies': ['casimir effect', 'quantum optics', 'vacuum engineering'],
                        'success_criteria': ['Stable negative energy production', 'Energy density measurement']
                    },
                    {
                        'name': 'Power Distribution System',
                        'duration': '18 months',
                        'dependencies': ['Negative Energy Generation'],
                        'key_technologies': ['energy distribution', 'power management', 'field coupling'],
                        'success_criteria': ['Efficient power delivery', 'System stability']
                    },
                    {
                        'name': 'Energy Storage Solutions',
                        'duration': '15 months',
                        'dependencies': ['Power Distribution System'],
                        'key_technologies': ['quantum storage', 'field containment', 'energy buffering'],
                        'success_criteria': ['Storage capacity targets', 'Retrieval efficiency']
                    }
                ]
            },
            'Phase 3: Field Generation': {
                'tasks': [
                    {
                        'name': 'Field Generator Design',
                        'duration': '30 months',
                        'dependencies': ['Spacetime Metric Engineering', 'Power Distribution System'],
                        'key_technologies': ['field coils', 'metric engineering', 'field geometry'],
                        'success_criteria': ['Generator prototype', 'Field stability metrics']
                    },
                    {
                        'name': 'Containment System Development',
                        'duration': '24 months',
                        'dependencies': ['Field Generator Design'],
                        'key_technologies': ['magnetic containment', 'field stabilization', 'geometric control'],
                        'success_criteria': ['Containment stability', 'Field maintenance']
                    },
                    {
                        'name': 'Field Control Interface',
                        'duration': '18 months',
                        'dependencies': ['Field Generator Design', 'Containment System Development'],
                        'key_technologies': ['control systems', 'feedback loops', 'field monitoring'],
                        'success_criteria': ['Control precision', 'Response time']
                    }
                ]
            },
            'Phase 4: Prototype Development': {
                'tasks': [
                    {
                        'name': 'Small-Scale Prototype',
                        'duration': '36 months',
                        'dependencies': ['Field Generator Design', 'Containment System Development', 'Energy Storage Solutions'],
                        'key_technologies': ['system integration', 'miniaturization', 'testing protocols'],
                        'success_criteria': ['Working prototype', 'Performance metrics']
                    },
                    {
                        'name': 'Scaling Analysis',
                        'duration': '12 months',
                        'dependencies': ['Small-Scale Prototype'],
                        'key_technologies': ['scale modeling', 'power scaling', 'field scaling'],
                        'success_criteria': ['Scaling models', 'Resource requirements']
                    },
                    {
                        'name': 'Safety Systems',
                        'duration': '24 months',
                        'dependencies': ['Small-Scale Prototype'],
                        'key_technologies': ['safety protocols', 'emergency systems', 'containment failure'],
                        'success_criteria': ['Safety certification', 'Emergency protocols']
                    }
                ]
            }
        }
        
    def _get_latest_data_dir(self) -> str:
        """Get the most recent data directory"""
        data_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('scrape_')]
        if not data_dirs:
            raise ValueError("No data directories found")
        return os.path.join(self.data_dir, sorted(data_dirs)[-1])

    def generate_task_network(self) -> None:
        """Generate a network visualization of task dependencies"""
        G = nx.DiGraph()
        
        # Add nodes and edges
        for phase, details in self.development_phases.items():
            for task in details['tasks']:
                G.add_node(task['name'], phase=phase)
                for dep in task['dependencies']:
                    G.add_edge(dep, task['name'])
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        for phase in self.development_phases.keys():
            phase_nodes = [node for node, attr in G.nodes(data=True) if attr['phase'] == phase]
            nx.draw_networkx_nodes(G, pos, nodelist=phase_nodes, node_size=2000, 
                                 node_color=f'C{list(self.development_phases.keys()).index(phase)}',
                                 alpha=0.6)
        
        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title("Warp Drive Development Task Dependencies")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, 'task_network.png'), bbox_inches='tight')
        plt.close()

    def generate_gantt_chart(self) -> None:
        """Generate a Gantt chart of the development timeline"""
        tasks = []
        current_start = 0
        
        for phase, details in self.development_phases.items():
            for task in details['tasks']:
                # Convert duration to months
                duration = int(task['duration'].split()[0])
                
                # Calculate start based on dependencies
                task_start = current_start
                if task['dependencies']:
                    # Find the latest end time of dependencies
                    dep_end_times = []
                    for dep in task['dependencies']:
                        for prev_task in tasks:
                            if prev_task['Task'] == dep:
                                dep_end_times.append(prev_task['Finish'])
                    if dep_end_times:
                        task_start = max(dep_end_times)
                
                tasks.append({
                    'Task': task['name'],
                    'Start': task_start,
                    'Finish': task_start + duration,
                    'Resource': phase
                })
        
        # Create Gantt chart
        df = pd.DataFrame(tasks)
        
        fig = ff.create_gantt(df,
                            colors=dict(zip(self.development_phases.keys(), 
                                          ['rgb(46, 137, 205)', 'rgb(114, 44, 121)', 
                                           'rgb(198, 47, 105)', 'rgb(58, 149, 136)'])),
                            index_col='Resource',
                            show_colorbar=True,
                            group_tasks=True,
                            showgrid_x=True,
                            showgrid_y=True)
        
        fig.update_layout(
            title="Warp Drive Development Timeline",
            xaxis_title="Months",
            height=800
        )
        
        fig.write_html(os.path.join(self.output_dir, 'development_timeline.html'))

    def generate_development_roadmap(self) -> str:
        """Generate detailed development roadmap report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate visualizations
        self.generate_task_network()
        self.generate_gantt_chart()
        
        report = f"""# Warp Drive Development Roadmap
Generated: {timestamp}

## Overview
This roadmap outlines the systematic development path towards achieving warp drive technology,
breaking down the process into manageable phases and specific tasks.

## Development Phases

"""
        # Add phase details
        for phase, details in self.development_phases.items():
            report += f"\n### {phase}\n\n"
            
            for task in details['tasks']:
                report += f"#### {task['name']}\n"
                report += f"- **Duration:** {task['duration']}\n"
                report += f"- **Dependencies:** {', '.join(task['dependencies']) if task['dependencies'] else 'None'}\n"
                report += "\nKey Technologies:\n"
                for tech in task['key_technologies']:
                    report += f"- {tech}\n"
                report += "\nSuccess Criteria:\n"
                for criterion in task['success_criteria']:
                    report += f"- {criterion}\n"
                report += "\n"

        # Add timeline analysis
        total_duration = sum(int(task['duration'].split()[0]) 
                           for phase in self.development_phases.values() 
                           for task in phase['tasks'])
        
        report += f"""
## Timeline Analysis

- Total Development Duration: {total_duration} months ({total_duration/12:.1f} years)
- Critical Path Activities:
  1. Theoretical Foundation Development
  2. Negative Energy Generation Systems
  3. Field Generator Design and Testing
  4. Prototype Development and Validation

## Risk Assessment and Mitigation

### Technical Risks
1. Negative energy generation stability
2. Field containment reliability
3. Power system scaling
4. Integration challenges

### Mitigation Strategies
1. Parallel development paths
2. Incremental testing approach
3. Redundant systems design
4. Comprehensive simulation program

## Resource Requirements

### Infrastructure
1. Quantum research facilities
2. Field generation laboratories
3. Power systems test facilities
4. Integration and testing centers

### Expertise
1. Theoretical physicists
2. Quantum engineers
3. Field specialists
4. Power systems engineers
5. Integration experts

## Success Metrics

### Phase Gates
1. Theoretical model validation
2. Energy system demonstration
3. Field generation stability
4. Prototype performance

### Performance Targets
1. Energy density thresholds
2. Field stability duration
3. Power efficiency metrics
4. System response times

## Next Steps

### Immediate Actions
1. Establish research teams
2. Set up core facilities
3. Begin theoretical modeling
4. Develop simulation frameworks

### Long-term Planning
1. Resource allocation strategy
2. Technology transfer plans
3. Scale-up methodology
4. Integration roadmap

## Visualization References
- Task dependency network: task_network.png
- Development timeline: development_timeline.html
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'development_roadmap.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report_path

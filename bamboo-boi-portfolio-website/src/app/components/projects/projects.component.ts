import { Component } from '@angular/core';

@Component({
  selector: 'app-projects',
  templateUrl: './projects.component.html',
  styleUrls: ['./projects.component.css']
})
export class ProjectsComponent {
  projects = [
    {
      title: 'Bamboo Tracker',
      description: 'A web application for tracking bamboo inventory and consumption for pandas around the world.',
      technologies: ['React', 'Node.js', 'MongoDB'],
      imageUrl: 'https://i.imgur.com/placeholder.png',
      link: '#'
    },
    {
      title: 'PandaChat',
      description: 'A real-time messaging platform for pandas to connect and share bamboo recipes with each other.',
      technologies: ['Vue.js', 'Firebase', 'Tailwind CSS'],
      imageUrl: 'https://i.imgur.com/placeholder.png',
      link: '#'
    },
    {
      title: 'BamboDB',
      description: 'A specialized database system optimized for storing and retrieving bamboo-related data efficiently.',
      technologies: ['Python', 'PostgreSQL', 'FastAPI'],
      imageUrl: 'https://i.imgur.com/placeholder.png',
      link: '#'
    }
  ];
} 
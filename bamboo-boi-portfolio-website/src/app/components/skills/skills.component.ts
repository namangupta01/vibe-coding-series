import { Component } from '@angular/core';

@Component({
  selector: 'app-skills',
  templateUrl: './skills.component.html',
  styleUrls: ['./skills.component.css']
})
export class SkillsComponent {
  skillCategories = [
    {
      name: 'Frontend',
      skills: ['HTML5', 'CSS3', 'JavaScript', 'TypeScript', 'React', 'Angular', 'Vue.js', 'Tailwind CSS', 'Bootstrap', 'SASS']
    },
    {
      name: 'Backend',
      skills: ['Node.js', 'Express', 'Python', 'Django', 'Flask', 'PHP', 'Laravel', 'Ruby on Rails', 'Java', 'Spring Boot']
    },
    {
      name: 'Databases',
      skills: ['MongoDB', 'PostgreSQL', 'MySQL', 'SQLite', 'Redis', 'Firebase']
    },
    {
      name: 'Other',
      skills: ['Git', 'Docker', 'Kubernetes', 'AWS', 'Google Cloud', 'CI/CD', 'Jest', 'Cypress', 'RESTful APIs', 'GraphQL']
    },
    {
      name: 'Panda Skills',
      skills: ['Professional Napping', 'Bamboo Munching', 'Slow-Motion Coding', 'Zen Meditation', 'Tree Climbing']
    }
  ];
}
 
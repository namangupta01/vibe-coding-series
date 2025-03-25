import { Component } from '@angular/core';

@Component({
  selector: 'app-hero',
  templateUrl: './hero.component.html',
  styleUrls: ['./hero.component.css']
})
export class HeroComponent {
  // Properties for hero section
  name = 'Bamboo Boi 🐼';
  title = 'Full Stack Developer & Bamboo Enthusiast';
  bio = '🌿 Hello there! I\'m a coding panda with serious development skills (and a serious bamboo addiction). 💻 I specialize in crafting delightful web experiences during my few waking hours between naps. 😴 My work schedule? Code for 2 hours, eat bamboo for 3, nap for 4, repeat! When I\'m not pushing commits, you\'ll find me rolling down hills or practicing the ancient panda art of looking cute while doing absolutely nothing.';

  // Fun facts about the panda developer
  pandaFacts = [
    "I type with my paws at 120 WPM (Words Per Munch)",
    "My debugging technique involves staring blankly at the screen (works every time!)",
    "I've never missed a deadline... according to my timezone (GMT-Panda)",
    "My code is as clean as my black and white fur"
  ];

  // Animated panda image URL - updating to a clearer panda GIF
  profileImage = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5_wCmhmZnBaENC7ygNneyAGYpa1_5PFduJg&s";

  scrollToProjects(): void {
    // In a real application, you might want to use ViewportScroller
    const projectsSection = document.getElementById('projects');
    if (projectsSection) {
      projectsSection.scrollIntoView({ behavior: 'smooth' });
    }
  }
  
  scrollToContact(): void {
    // In a real application, you might want to use ViewportScroller
    const contactSection = document.getElementById('contact');
    if (contactSection) {
      contactSection.scrollIntoView({ behavior: 'smooth' });
    }
  }
} 
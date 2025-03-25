import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'Bamboo Boi - Full Stack Developer';
  
  constructor() {
    this.initializeTheme();
  }
  
  initializeTheme(): void {
    // Check for user preference in local storage
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    
    // Set initial mode based on user preference or system preference
    if (isDarkMode || (localStorage.getItem('darkMode') === null && 
        window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }
} 
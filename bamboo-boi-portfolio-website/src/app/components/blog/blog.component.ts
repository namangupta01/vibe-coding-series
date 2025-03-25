import { Component } from '@angular/core';

@Component({
  selector: 'app-blog',
  templateUrl: './blog.component.html',
  styleUrls: ['./blog.component.css']
})
export class BlogComponent {
  blogPosts = [
    {
      id: 1,
      title: "My First Day as a Developer 🖥️",
      date: "March 15, 2023",
      image: "https://media.giphy.com/media/RIqKz4BjsgLSg/giphy.gif",
      excerpt: "Today was my first day as a developer! I was so excited I could barely finish my morning bamboo...",
      content: `
        <p>🌿 Dear Diary (and fellow pandas),</p>
        
        <p>Today was my FIRST day as a professional developer! I was so excited I could barely finish my morning bamboo (and that's saying something – I NEVER skip bamboo).</p>
        
        <p>The humans set up my workstation with THREE monitors! Can you believe it? I only have two eyes! 👀 They also gave me this fancy ergonomic chair that can support up to 500 pounds, which is perfect for post-lunch food comas.</p>
        
        <p>I spent most of the morning trying to type with my paws. Pro tip for other panda developers: press one key at a time and invest in a REALLY big keyboard. My team lead kept saying something about "pair programming," but I think he just wanted to take over when my claws got stuck between the keys.</p>
        
        <p>Lunch break was amazing - they have a bamboo snack bar! I think it was actually a regular snack bar, but I brought my own bamboo and no one complained.</p>
        
        <p>By the end of the day, I had written my first "Hello World" program only 17 times slower than my human colleagues. Progress! 🎉</p>
        
        <p>Tomorrow's goal: Figure out how to stop falling asleep during stand-up meetings.</p>
        
        <p>Rolling back to my bamboo grove,<br>Bamboo Boi 🐼</p>
      `
    },
    {
      id: 2,
      title: "The Great Git Disaster of Tuesday 😱",
      date: "March 16, 2023",
      image: "https://media2.giphy.com/media/EtB1yylKGGAUg/giphy.gif",
      excerpt: "Note to self: 'git push --force' is NOT the solution to every problem...",
      content: `
        <p>🌿 Oh my bamboo shoots,</p>
        
        <p>What a day! I learned about Git today, and let me tell you, it was NOT as delicious as it sounds. 🍽️</p>
        
        <p>The morning started peacefully enough. I was assigned to a small task - just updating a README file. "Even a sleepy panda can do this," I thought. WRONG.</p>
        
        <p>I made my changes, then needed to "commit" them (which is apparently not the same as committing to a nap schedule). I typed a few commands, got some angry red text, panicked, and then remembered that one of the developers yesterday mentioned something called "force push."</p>
        
        <p>Well, let me tell you what happens when you "git push --force" to the main branch... 🔥</p>
        
        <p>The humans made these strange high-pitched noises and gathered around my desk in what I can only describe as a panic circle. Words like "revert," "backup," and "WHY???" were thrown around.</p>
        
        <p>Three emergency pizzas and seven hours later, we managed to recover most of the code. My team lead took a deep breath and said, "This is why we have code reviews."</p>
        
        <p>I've been assigned Git tutorials for the rest of the week. Also, they revoked my push access. Probably for the best.</p>
        
        <p>Hiding in my virtual bamboo forest,<br>Bamboo Boi 🐼</p>
      `
    },
    {
      id: 3,
      title: "How I Solved a Bug During My Nap 💤",
      date: "March 20, 2023",
      image: "https://media.giphy.com/media/13YKhR4Iw9HQRi/giphy.gif",
      excerpt: "Turns out my best debugging happens when I'm asleep! Here's how my subconscious fixed a memory leak...",
      content: `
        <p>🌿 Fellow code bamboo chewers,</p>
        
        <p>BREAKTHROUGH! I've discovered my secret weapon for debugging: napping! 😴</p>
        
        <p>We had this persistent memory leak in the application that nobody could figure out. I spent all morning staring at the code until my eyes crossed and I started seeing double bamboo.</p>
        
        <p>Around lunchtime, after my usual 3 pounds of bamboo shoots, I felt the familiar food coma setting in. I explained to my team that I needed to "process the problem in parallel" and proceeded to curl up under my desk for my standard 2-hour nap.</p>
        
        <p>And you won't believe what happened! In my dream, I was swimming through lines of code (weird, I know), when I spotted it - we were creating new subscriptions but never unsubscribing! The memory leak was right there, disguised as a floating bamboo shoot that never got eaten!</p>
        
        <p>I woke up with a jolt, rolled back to my computer (literally rolled, it's how pandas move efficiently), and fixed the issue in 5 minutes. The team was so impressed!</p>
        
        <p>My manager has now officially approved "debug naps" as part of my workflow. I think I'm revolutionizing panda programming practices!</p>
        
        <p>Dreamily yours,<br>Bamboo Boi 🐼</p>
        
        <p>P.S. If you're struggling with a coding problem, I highly recommend trying the "sleep on it" approach. Works like a charm!</p>
      `
    }
  ];
  
  activePost: any = null;
  
  showPostDetails(post: any): void {
    this.activePost = post;
    // In a real app, you might navigate to a dedicated post page
    // For simplicity, we're just showing the full content in the same page
    setTimeout(() => {
      const element = document.getElementById('post-details');
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  }
  
  goBack(): void {
    this.activePost = null;
    setTimeout(() => {
      const element = document.getElementById('blog');
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  }
} 
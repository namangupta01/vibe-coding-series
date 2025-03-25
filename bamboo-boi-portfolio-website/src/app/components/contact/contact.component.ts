import { Component } from '@angular/core';

@Component({
  selector: 'app-contact',
  templateUrl: './contact.component.html',
  styleUrls: ['./contact.component.css']
})
export class ContactComponent {
  contactInfo = {
    email: 'bamboo.boi@pandadev.com',
    phone: '+1 (555) 123-4567',
    location: 'Bamboo Forest, Eastern Mountains',
    socialLinks: [
      { platform: 'GitHub', url: '#', icon: 'github' },
      { platform: 'Instagram', url: '#', icon: 'instagram' },
      { platform: 'Twitter', url: '#', icon: 'twitter' },
      { platform: 'LinkedIn', url: '#', icon: 'linkedin' }
    ]
  };

  submitForm(event: Event): void {
    event.preventDefault();
    // In a real application, we would implement form submission logic here
    console.log('Form submitted');
    alert('Thank you for your message! I will get back to you soon.');
  }
} 
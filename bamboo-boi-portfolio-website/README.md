# Bamboo Boi - Angular Portfolio Website

A responsive Angular portfolio website for a software developer named "Bamboo Boi" (a panda). This website showcases the developer's skills, projects, and contact information.

## Features

- **Built with Angular**: Utilizes Angular framework for component-based structure
- **Responsive Design**: Works on all devices from mobile to desktop
- **Dark Mode**: Toggle between light and dark themes
- **Smooth Scrolling**: For internal navigation links
- **Modern UI**: Clean and professional design with animations
- **Accessibility**: Focus states and semantic HTML
- **Mobile-Friendly Navigation**: Collapsible menu on smaller screens
- **Tailwind CSS**: Utility-first CSS framework via CDN

## Technologies Used

- **Angular 16**: Component-based structure and routing
- **TypeScript**: For type-safe code
- **HTML5**: Semantic markup
- **CSS3**: Custom styling with animations
- **Tailwind CSS**: Via CDN for utility-first styling
- **SVG Icons**: For clean, scalable iconography

## Project Structure

- **Components**:
  - Header: Navigation and dark mode toggle
  - Hero: Introduction section
  - Projects: Portfolio projects in card layout
  - Skills: Developer skills categorized
  - Contact: Contact form and information
  - Footer: Site footer with navigation links

## Setup and Usage

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. Build for production:
   ```
   npm run build
   ```

## Component Modules

The website uses Angular's lazy loading for better performance:

- Hero Module: Displays the about/intro section
- Projects Module: Showcases portfolio projects
- Skills Module: Presents developer skills by category
- Contact Module: Provides contact form and info

## Dark Mode Implementation

The website implements dark mode using Tailwind CSS's dark mode with class strategy:
- Dark mode preference is saved to localStorage
- It respects the user's system preferences initially
- A toggle button in the header allows switching between modes

## Responsive Design

- Mobile-first approach with Tailwind CSS breakpoints
- Custom hamburger menu for mobile devices
- Flexible grid layouts for different screen sizes
- Optimized typography and spacing for all devices

## Customization

- **Content**: Edit the component files to customize content
- **Styling**: Modify the CSS files or use Tailwind classes
- **Colors**: Update the Tailwind configuration in index.html
- **Projects**: Update the projects array in the projects component
- **Skills**: Modify the skills categories in the skills component

## Future Enhancements

- Add animations and transitions between routes
- Implement form validation and submission
- Add a blog section
- Include more interactive elements
- Optimize for improved performance

## License

MIT

## Author

Bamboo Boi - A coding panda with a passion for web development and bamboo. 
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [typedText, setTypedText] = useState('');
  const [typingIndex, setTypingIndex] = useState(0);
  const [typingComplete, setTypingComplete] = useState(false);
  const [showOutput, setShowOutput] = useState(false);

  const sections = useRef({
    home: null,
    about: null,
    projects: null,
    skills: null,
    experience: null,
    education: null,
    certifications: null,
    contact: null
  });

  // Typing animation effect
  useEffect(() => {
    const fullText = "print('Hello World!')";
    if (typingIndex < fullText.length) {
      const timer = setTimeout(() => {
        setTypedText(fullText.substring(0, typingIndex + 1));
        setTypingIndex(typingIndex + 1);
      }, 100);
      return () => clearTimeout(timer);
    } else {
      setTypingComplete(true);
      const outputTimer = setTimeout(() => {
        setShowOutput(true);
      }, 500);
      return () => clearTimeout(outputTimer);
    }
  }, [typingIndex]);

  useEffect(() => {
    setupIntersectionObserver();
  }, []);

  const setupIntersectionObserver = () => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.5 }
    );

    Object.values(sections.current).forEach(section => {
      if (section) observer.observe(section);
    });
  };

  const scrollToSection = (sectionId) => {
    sections.current[sectionId]?.scrollIntoView({ behavior: 'smooth' });
    setIsMenuOpen(false);
  };

  // UPDATED: handle form submission
  const handleFormSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    try {
        // Change the URL to a relative path. The proxy will handle the rest.
        const response = await fetch('/send-email/', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (response.ok) {
            alert('Message sent successfully! I will get back to you soon.');
            event.target.reset();
        } else {
            alert('Failed to send message. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again later.');
    }
};

  // Sample data (unchanged)
  const profileData = {
    name: "Yashwanth R",
    title: "AI & Machine Learning Engineer",
    bio: "Building intelligent systems that learn and adapt. Passionately about deep learning, computer vision, and creating AI that makes a difference.",
    email: "yashwanth150204@gmail.com",
    phone: "+91 9345570940",
    location: "Coimbatore, India",
    social: {
      github: "https://github.com/Yashwanth1524",
      linkedin: "https://linkedin.com/in/yashwanth1524",
      instagram: "https://www.instagram.com/yashwanth1524?igsh=MWkyd2JnZXVvZm83eg=="
    }
  };

  const projects = [
    {
      id: 1,
      title: "Optical Music Recognition (OMR) System",
      description: "Developed an AI-based system to automatically recognize and digitize musical notation from sheet music.",
      technologies: ["Python", "TensorFlow", "CNN", "OpenCV"],
      github: "https://github.com/Yashwanth1524/OMR"
    },
    {
      id: 2,
      title: "Denoising Musical Sheet",
      description: "An API using FastAPI and OpenCV to clean and enhance noisy or damaged musical sheet images for improved recognition.",
      technologies: ["Python", "FastAPI", "OpenCV"],
      github: "https://github.com/Yashwanth1524/OMR",
      live: "http://127.0.0.1:8000/denoise-demo/"
    },
    {
      id: 3,
      title: "E-commerce Application with Servlets",
      description: "A comprehensive e-commerce platform built on Java servlets for server-side logic and database interaction.",
      technologies: ["Java", "Servlets", "JSP", "HTML/CSS", "MySQL"],
      github: "https://github.com/Yashwanth1524/E-commerce-Application-with-Servlets"
    },
    {
      id: 4,
      title: "Blog Website with ReactJS and Firebase",
      description: "A dynamic blog website with user authentication and real-time content management, powered by React and Firebase.",
      technologies: ["React", "Firebase", "HTML/CSS"],
      github: "https://github.com/Yashwanth1524/socio"
    }
  ];

  const skills = {
    "Machine Learning": {
      description: "Used libraries like TensorFlow and OpenCV to build intelligent systems.",
      items: ["TensorFlow", "PyTorch", "Scikit-learn", "OpenCV"]
    },
    "Deep Learning": {
      description: "Created neural networks like CNNs for advanced tasks such as music recognition.",
      items: ["CNN", "RNN", "Transformers", "GANs"]
    },
    "Programming": {
      description: "Developed projects using multiple languages to build scalable and efficient solutions.",
      items: ["Python", "Java", "JavaScript"]
    },
    "Web Development": {
      description: "Built responsive and dynamic web applications with modern frameworks.",
      items: ["React", "Node.js", "HTML/CSS", "Firebase"]
    }
  };

  const experience = [
    {
      company: "University of Malaya, Institute for Advanced Studies, Malaysia",
      position: "Research Intern - AI for Music Digitization",
      period: "Jul 2024 - Oct 2024",
      details: [
        "Developed AI-based Optical Music Recognition (OMR) system achieving 92% accuracy in digitizing historical musical sheets",
        "Implemented hybrid denoising algorithm combining CNN and FASTNLMEANS image processing, reducing noise by 40% in aged documents",
        "Designed segmentation pipeline using OpenCV and TensorFlow, improving staff line detection accuracy by 35%",
        "Integrated ABC notation converter for standardized digital output, enabling compatibility with 15+ music analysis tools",
        "Collaborated with musicologists to validate results on Primus dataset containing 10,000+ musical samples"
      ]
    }
  ];

  const education = [
    {
      institution: "Coimbatore Institute of Technology",
      degree: "M.Sc. Artificial Intelligence and Machine Learning",
      period: "2021 - 2026",
      details: "CGPA: 7.89"
    }
  ];

  const certifications = [
    {
      name: "Mobile Application Development",
      issuer: "PSG College of Technology",
      date: "March 24-26, 2023",
      link: "https://drive.google.com/file/d/1AUZFeQx_NZ6PSob42xjZOiiFYnCbUEcK/view?usp=drive_link"
    },
    {
      name: "Melinia Hackathon 2k24",
      issuer: "Coimbatore Institute of Technology",
      date: "February 2024",
      link: "https://drive.google.com/file/d/1APbQ_Y7L2NWRvUJA2zGECOitTKcwO3zF/view?usp=drive_link"
    },
    {
      name: "GDSC - Summer Hackathon 2023",
      issuer: "Google Developer Student Clubs, CIT",
      date: "March 18, 2023",
      link: "https://drive.google.com/file/d/1AHCIhAyE4gJbcwyCqyUbP45aPB1ypqfR/view?usp=drive_link"
    },
    {
      name: "Diploma in Programming Languages",
      issuer: "Career Surfing Gateway (CSG)",
      date: "Oct 2022 - Feb 2023",
      link: "https://drive.google.com/file/d/1ASGh8MlSYHmbICrEumn18_CKHNy3XFQV/view?usp=drive_link"
    },
    {
      name: "Cryptic Quest at Melinia'24",
      issuer: "Coimbatore Institute of Technology",
      date: "February 16, 2024",
      link: "https://drive.google.com/file/d/1AD6aJoF8m_TAAF6ZqYf5ZTILp-0zu0od/view?usp=drive_link"
    },
    {
      name: "Techfusion-X at Ersmeronz'24",
      issuer: "Bannari Amman Institute of Technology",
      date: "March 1-2, 2024",
      link: "https://drive.google.com/file/d/19tf3Q-llzacK_oqiOjJMQ0VIprSYH1iC/view?usp=drive_link"
    },
    {
      name: "Cryptera 2024 Hackathon",
      issuer: "Coimbatore Institute of Technology",
      date: "March 1-2, 2024",
      link: "https://drive.google.com/file/d/1AO5wYQ_jC0iyiCr35ZyiQoXSN39Tt-w-/view?usp=drive_link"
    },
    {
      name: "Internship Completion",
      issuer: "University Of Malaysia",
      date: "July 2024 - October 2024",
      link: "https://drive.google.com/file/d/1usVP6Dtq3Fe_cBcgo3MaJ90_grZSEcYB/view?usp=drive_link"
    }
  ];

  return (
    <div className="App">
      {/* Animated background elements (unchanged) */}
      <div className="animated-background">
        <div className="floating-shapes">
          <div className="shape shape-1"></div>
          <div className="shape shape-2"></div>
          <div className="shape shape-3"></div>
        </div>
        <div className="binary-rain"></div>
      </div>

      {/* Navigation (unchanged) */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">
            <span className="logo-icon">{'</>'}</span>
            <span>Yashwanth</span>
          </div>
          <div className={`nav-menu ${isMenuOpen ? 'active' : ''}`}>
            {['home', 'about', 'projects', 'skills', 'experience', 'education', 'certifications', 'contact'].map((item) => (
              <button
                key={item}
                className={`nav-link ${activeSection === item ? 'active' : ''}`}
                onClick={() => scrollToSection(item)}
              >
                {item.charAt(0).toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
          <button className="nav-toggle" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            <span></span>
            <span></span>
            <span></span>
          </button>
        </div>
      </nav>

      {/* Hero Section (unchanged) */}
      <section id="home" className="hero" ref={el => sections.current.home = el}>
        <div className="hero-content">
          <div className="hero-text">
            <div className="hero-badge">AI/ML Engineer</div>
            <h1 className="hero-title">
              Transforming Ideas into <span className="text-gradient">Intelligent Solutions</span>
            </h1>
            <p className="hero-description">
              I create cutting-edge AI systems that learn, adapt, and solve real-world problems. 
              Passionately about deep learning, computer vision, and innovative technology.
            </p>
            <div className="hero-buttons">
              <button className="btn btn-primary" onClick={() => scrollToSection('projects')}>
                <span>View My Work</span>
                <div className="btn-hover-effect"></div>
              </button>
              <button className="btn btn-primary" onClick={() => scrollToSection('contact')}>
                <span>Get In Touch</span>
                <div className="btn-hover-effect"></div>
              </button>
              <a href="/resume.pdf" target="_blank" rel="noopener noreferrer" className="btn btn-primary no-underline">
                <span>View Resume</span>
                <div className="btn-hover-effect"></div>
              </a>
            </div>
            <div className="hero-stats">
              <div className="stat">
                <span className="stat-number">5+</span>
                <span className="stat-label">Projects</span>
              </div>
              <div className="stat">
                <span className="stat-number">15+</span>
                <span className="stat-label">Technologies</span>
              </div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="floating-card">
              <div className="terminal-animation">
                <div className="terminal-header">
                  <div className="terminal-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <div className="terminal-title">bash — 80×30</div>
                </div>
                <div className="terminal-body">
                  <div className="terminal-line">
                    <span className="terminal-prompt">yashwanth@portfolio:~$ </span>
                    <span className="terminal-command">{typedText}</span>
                    <span className={`terminal-cursor ${!typingComplete ? 'blinking' : ''}`}>|</span>
                  </div>
                  {typingComplete && showOutput && (
                    <>
                      <div className="terminal-line">
                        <span className="terminal-output">Hello World!</span>
                      </div>
                      <div className="terminal-line">
                        <span className="terminal-output">I am Yashwanth, an AI/ML Engineer.</span>
                      </div>
                      <div className="terminal-line">
                        <span className="terminal-prompt">yashwanth@portfolio:~$ </span>
                        <span className={`terminal-cursor blinking`}>|</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="scroll-indicator">
          <div className="scroll-arrow"></div>
        </div>
      </section>

      {/* About Section (unchanged) */}
      <section id="about" className="section" ref={el => sections.current.about = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">About Me</h2>
            <p className="section-subtitle">Get to know me better</p>
          </div>
          <div className="about-grid">
            <div className="about-content">
              <p>
                I'm a passionate AI and Machine Learning student with expertise in developing 
                intelligent systems. My work focuses on deep learning, computer vision, and 
                creating practical AI solutions that solve real-world problems.
              </p>
              <p>
                With experience in research and development, I've worked on projects ranging from 
                music digitization to predictive maintenance systems. I thrive on challenges and enjoy 
                pushing the boundaries of what's possible with AI.
              </p>
            </div>
            <div className="about-visual">
              <div className="spline-robot-animation">
                <iframe 
                  src='https://my.spline.design/genkubgreetingrobot-39YvET0DHmkK3lJ9xG0QbXbu/' 
                  frameBorder='0'
                  width='100%'
                  height='100%'
                  title="3D Robot Animation"
                ></iframe>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Projects Section (unchanged) */}
      <section id="projects" className="section" ref={el => sections.current.projects = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Featured Projects</h2>
            <p className="section-subtitle">My latest creative work</p>
          </div>
          <div className="projects-grid">
            {projects.map((project, index) => (
              <div key={project.id} className="project-card" data-index={index}>
                <div className="project-header">
                  <h3>{project.title}</h3>
                  <div className="project-links">
                    <a href={project.github} target="_blank" rel="noopener noreferrer" className="project-link">
                      <span className="project-link-icon">📁</span>
                      <span>GitHub</span>
                    </a>
                    {project.live && (
                      <a href={project.live} target="_blank" rel="noopener noreferrer" className="project-link">
                        <span className="project-link-icon">🌐</span>
                        <span>Live Demo</span>
                      </a>
                    )}
                  </div>
                </div>
                <p className="project-description">{project.description}</p>
                <div className="project-technologies">
                  {project.technologies.map(tech => (
                    <span key={tech} className="tech-tag">{tech}</span>
                  ))}
                </div>
                <div className="project-hover-effect"></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Skills Section (unchanged) */}
      <section id="skills" className="section" ref={el => sections.current.skills = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Technical Skills</h2>
            <p className="section-subtitle">Technologies I work with</p>
          </div>
          <div className="skills-container">
            {Object.entries(skills).map(([category, data]) => (
              <div key={category} className="skill-category">
                <div className="skill-category-header">
                  <h3 className="skill-category-title">{category}</h3>
                  <p className="skill-category-description">{data.description}</p>
                </div>
                <div className="skills-list-wrapper">
                  <div className="skills-grid">
                    {data.items.map(skill => (
                      <div key={skill} className="skill-item-no-progress">
                        <span className="skill-name">{skill}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Experience Section (unchanged) */}
      <section id="experience" className="section" ref={el => sections.current.experience = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Work Experience</h2>
            <p className="section-subtitle">My professional journey</p>
          </div>
          <div className="timeline-grid">
            <div className="timeline-content-container">
              {experience.map((exp, index) => (
                <div key={index} className="experience-card">
                  <h3>{exp.position}</h3>
                  <h4>{exp.company}</h4>
                  <span className="experience-period">{exp.period}</span>
                  <ul className="experience-details">
                    {exp.details.map((detail, i) => (
                      <li key={i}>{detail}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <div className="timeline-image-container">
              <img src="/um.jpg" alt="University of Malaya" className="timeline-image" />
              <img src="/um2.jpg" alt="University of Malaya Campus" className="timeline-image" />
            </div>
          </div>
        </div>
      </section>

      {/* Education Section (unchanged) */}
      <section id="education" className="section" ref={el => sections.current.education = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Education</h2>
            <p className="section-subtitle">My academic background</p>
          </div>
          <div className="education-grid">
            {education.map((edu, index) => (
              <div key={index} className="education-card">
                <div className="education-icon">🎓</div>
                <h3>{edu.degree}</h3>
                <h4>{edu.institution}</h4>
                <span className="education-period">{edu.period}</span>
                <p>{edu.details}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Certifications Section (unchanged) */}
      <section id="certifications" className="section" ref={el => sections.current.certifications = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Certifications</h2>
            <p className="section-subtitle">My professional training</p>
          </div>
          <div className="certifications-grid">
            {certifications.map((cert, index) => (
              <a href={cert.link} key={index} target="_blank" rel="noopener noreferrer" className="certification-card">
                <div className="certification-icon">🏅</div>
                <h3>{cert.name}</h3>
                <p><strong>Issuer:</strong> {cert.issuer}</p>
                <p><strong>Date:</strong> {cert.date}</p>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section (updated) */}
      <section id="contact" className="section" ref={el => sections.current.contact = el}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Get In Touch</h2>
            <p className="section-subtitle">Let's work together</p>
          </div>
          <div className="contact-grid">
            <div className="contact-info">
              <div className="contact-card">
                <h3>Ready to start your project?</h3>
                <p>I'm available for freelance work and new opportunities. Let's discuss how I can help bring your ideas to life.</p>
                <div className="contact-details">
                  <div className="contact-item">
                    <div className="contact-icon">📧</div>
                    <div>
                      <div className="contact-label">Email</div>
                      <a href={`mailto:${profileData.email}`}>{profileData.email}</a>
                    </div>
                  </div>
                  <div className="contact-item">
                    <div className="contact-icon">📞</div>
                    <div>
                      <div className="contact-label">Phone</div>
                      <a href={`tel:${profileData.phone}`}>{profileData.phone}</a>
                    </div>
                  </div>
                  <div className="contact-item">
                    <div className="contact-icon">📍</div>
                    <div>
                      <div className="contact-label">Location</div>
                      <span>{profileData.location}</span>
                    </div>
                  </div>
                </div>
                <div className="social-links">
                  {Object.entries(profileData.social).map(([platform, url]) => (
                    <a key={platform} href={url} target="_blank" rel="noopener noreferrer" className="social-link">
                      <span className={`social-icon ${platform}`}></span>
                    </a>
                  ))}
                </div>
              </div>
            </div>
            <div className="contact-form-container">
              <h3>Send me a message</h3>
              {/* UPDATED form element */}
              <form className="contact-form" onSubmit={handleFormSubmit}>
                <div className="form-group">
                  <input type="text" name="name" placeholder="Your Name" required />
                </div>
                <div className="form-group">
                  <input type="email" name="email" placeholder="Your Email" required />
                </div>
                <div className="form-group">
                  <input type="text" name="subject" placeholder="Subject" required />
                </div>
                <div className="form-group">
                  <textarea name="body" placeholder="Your Message" rows="5" required></textarea>
                </div>
                <button type="submit" className="btn btn-primary">
                  <span>Send Message</span>
                  <div className="btn-hover-effect"></div>
                </button>
              </form>
            </div>
          </div>
        </div>
      </section>

      {/* Footer (unchanged) */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-logo">
              <span className="logo-icon">{'</>'}</span>
              <span>Yashwanth R</span>
            </div>
            <p>Building the future with AI and Machine Learning</p>
            <div className="footer-links">
              <a href="#home">Home</a>
              <a href="#about">About</a>
              <a href="#projects">Projects</a>
              <a href="#contact">Contact</a>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2024 Yashwanth R. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
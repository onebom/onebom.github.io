---
layout: default
title: about
permalink: /
---

<link rel="stylesheet" href="{{ '/assets/css/about.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/css/projects.css' | relative_url }}">
{% assign bio = site.data.about %}
<div class="academic-about">
  <header class="about-hero">
    <h1 class="about-name">{{ bio.name }}</h1>
    <div class="about-introduction">
      <div class="about-bio">{{ bio.intro | markdownify }}</div>
      <nav class="about-links" aria-label="Profile links">
        {% for link in bio.links %}
          {% if link.url and link.url != empty %}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% endif %}
        {% endfor %}
      </nav>
    </div>
    <div class="about-portrait">
      {% if bio.portrait != empty %}
        <img src="{{ bio.portrait | relative_url }}" alt="{{ bio.name }}" width="240" height="240">
      {% else %}
        <div class="portrait-initial" role="img" aria-label="onebom monogram">ob<span>onebom</span></div>
      {% endif %}
    </div>
  </header>
  <section class="about-section about-records" aria-labelledby="education-heading">
    <h2 id="education-heading">Education</h2>
    {% if bio.education.size > 0 %}
    <ul class="record-list">
      {% for item in bio.education %}
      <li>
        <div class="record-title markdown-content">{{ item.title | markdownify }}</div>
        {% if item.date %}<p class="record-date">{{ item.date }}</p>{% endif %}
        {% if item.role %}<div class="record-role markdown-content">{{ item.role | markdownify }}</div>{% endif %}
        {% if item.authors or item.name %}<div class="record-authors markdown-content">{{ item.authors | default: item.name | markdownify }}</div>{% endif %}
        {% if item.venue %}<p class="record-venue">{{ item.venue }}{% if item.year %}, {{ item.year }}{% endif %}</p>{% endif %}
        {% if item.description %}<div class="markdown-content">{{ item.description | markdownify }}</div>{% endif %}
        <div class="record-links">
          {% for link in item.links %}
            {% if link.url and link.url != empty %}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% endif %}
          {% endfor %}
        </div>
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
  <section class="about-section about-records" aria-labelledby="experience-heading">
    <h2 id="experience-heading">Experience</h2>
    {% if bio.experience.size > 0 %}
    <ul class="record-list">
      {% for item in bio.experience %}
      <li>
        <div class="record-title experience-title"><span>{{ item.title | markdownify | remove: '<p>' | remove: '</p>' }}</span>{% if item.date %} <span class="experience-date">({{ item.date }})</span>{% endif %}</div>
        {% if item.role %}<div class="record-role markdown-content">{{ item.role | markdownify }}</div>{% endif %}
        {% if item.authors or item.name %}<div class="record-authors markdown-content">{{ item.authors | default: item.name | markdownify }}</div>{% endif %}
        {% if item.venue %}<p class="record-venue">{{ item.venue }}{% if item.year %}, {{ item.year }}{% endif %}</p>{% endif %}
        {% if item.description %}<div class="markdown-content">{{ item.description | markdownify }}</div>{% endif %}
        <div class="record-links">
          {% for link in item.links %}
            {% if link.url and link.url != empty %}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% endif %}
          {% endfor %}
        </div>
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
  <section class="about-section about-records" aria-labelledby="publications-heading">
    <h2 id="publications-heading">Publications</h2>
    {% if bio.publications.size > 0 %}
    <div class="project-list publication-list">
      {% for project in bio.publications %}
  {% assign project_page_url = project.project_url | default: project.url %}
  {% if project_page_url == nil or project_page_url == empty %}
    {% for link in project.links %}{% assign label = link.label | downcase %}{% if label == 'project page' %}{% assign project_page_url = link.url %}{% endif %}{% endfor %}
    {% if project.link contains '[Project page](' %}{% assign project_page_url = project.link | split: '[Project page](' | last | split: ')' | first %}{% endif %}
  {% endif %}
  <article class="project-entry">
    <div class="project-thumbnail">
      {% if project_page_url and project_page_url != empty %}<a class="project-image-link" href="{{ project_page_url | escape }}" aria-label="View project: {{ project.title | strip_html | escape }}">{% endif %}
      {% if project.image and project.image != empty %}
      <img src="{{ project.image | relative_url }}" alt="{{ project.image_alt | default: project.title | escape }}" loading="lazy">
      {% else %}
      <div class="project-thumbnail-placeholder" aria-hidden="true">{% if project.authors or project.name %}Paper{% else %}{{ project.mark | default: 'Project' }}{% endif %}</div>
      {% endif %}
      {% if project_page_url and project_page_url != empty %}</a>{% endif %}
    </div>
    <div class="project-details">
      <h3>{% if project_page_url and project_page_url != empty %}<a class="project-title-link" href="{{ project_page_url | escape }}">{% endif %}{{ project.title | markdownify | remove: '<p>' | remove: '</p>' }}{% if project_page_url and project_page_url != empty %}</a>{% endif %}</h3>
      {% if project.authors or project.name %}<div class="record-authors markdown-content">{{ project.authors | default: project.name | markdownify }}</div>{% endif %}
      {% if project.venue or project.year or project.date %}
      <p class="project-venue">{% if project.venue %}<em>{{ project.venue | escape }}</em>{% endif %}{% if project.year %}{% if project.venue %}, {% endif %}{{ project.year }}{% elsif project.date %}{% if project.venue %}, {% endif %}{{ project.date }}{% endif %}</p>
      {% endif %}
      <div class="project-links">
        {% if project.link and project.link != empty %}
          {{ project.link | markdownify }}
        {% elsif project.links.size > 0 %}
          {% assign separator = '' %}{% for link in project.links %}{% if link.url and link.url != empty %}{{ separator }}<a href="{{ link.url | escape }}">{{ link.label | escape }}</a>{% assign separator = ' / ' %}{% endif %}{% endfor %}
        {% elsif project.url %}
          <a href="{{ project.url | escape }}">Project page</a>
        {% endif %}
      </div>
      {% if project.description %}<div class="project-description">{{ project.description | markdownify }}</div>{% endif %}
    </div>
  </article>
      {% endfor %}
    </div>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
  <section class="about-section about-records" aria-labelledby="projects-heading">
    <h2 id="projects-heading">Projects</h2>
    {% if bio.projects.size > 0 %}
    <ul class="record-list about-project-records">
      {% for project in bio.projects %}
  {% assign project_page_url = project.project_url | default: project.url %}
  {% if project_page_url == nil or project_page_url == empty %}
    {% for link in project.links %}{% assign label = link.label | downcase %}{% if label == 'project page' %}{% assign project_page_url = link.url %}{% endif %}{% endfor %}
    {% if project.link contains '[Project page](' %}{% assign project_page_url = project.link | split: '[Project page](' | last | split: ')' | first %}{% endif %}
  {% endif %}
      {% assign project_date = project.date | default: project.year %}
      <li>
        <div class="record-title experience-title"><span>{% if project_page_url and project_page_url != empty %}<a class="project-title-link" href="{{ project_page_url | escape }}">{% endif %}{{ project.title | markdownify | remove: '<p>' | remove: '</p>' }}{% if project_page_url and project_page_url != empty %}</a>{% endif %}</span></div>
        {% if project.venue or project_date %}<div class="record-role markdown-content project-venue-date">{% if project.venue %}<span>{{ project.venue | markdownify | remove: '<p>' | remove: '</p>' }}</span>{% endif %}{% if project_date and project_date != empty %} <span>({{ project_date }})</span>{% endif %}</div>{% endif %}
        {% if project.role %}<div class="record-role markdown-content">{{ project.role | markdownify }}</div>{% endif %}
        {% if project.description and project.description != empty %}
          {% assign description_html = project.description | markdownify %}
          <div class="markdown-content">{% if description_html contains '<ul>' or description_html contains '<ol>' %}{{ description_html }}{% else %}<ul><li>{{ description_html }}</li></ul>{% endif %}</div>
        {% endif %}
        {% if project.link and project.link != empty %}
          {% assign project_link_html = project.link | markdownify %}
          <div class="markdown-content">{% if project_link_html contains '<ul>' or project_link_html contains '<ol>' %}{{ project_link_html }}{% else %}<ul><li>{{ project_link_html }}</li></ul>{% endif %}</div>
        {% elsif project.links.size > 0 %}
          {% capture project_link_items %}{% assign separator = '' %}{% for link in project.links %}{% if link.url and link.url != empty %}{{ separator }}<a href="{{ link.url | escape }}">{{ link.label | escape }}</a>{% assign separator = ' / ' %}{% endif %}{% endfor %}{% endcapture %}
          {% assign project_link_items = project_link_items | strip %}
          {% if project_link_items != empty %}<div class="markdown-content"><ul><li>{{ project_link_items }}</li></ul></div>{% endif %}
        {% elsif project.url %}
          <div class="markdown-content"><ul><li><a href="{{ project.url | escape }}">Project page</a></li></ul></div>
        {% endif %}
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
  <section class="about-section about-records" aria-labelledby="honors-heading">
    <h2 id="honors-heading">Honors and Awards</h2>
    {% if bio.honors.size > 0 %}
    <ul class="record-list">
      {% for item in bio.honors %}
      <li>
        <div class="record-title markdown-content">{{ item.title | markdownify }}</div>
        {% if item.date %}<p class="record-date">{{ item.date }}</p>{% endif %}
        {% if item.role %}<div class="record-role markdown-content">{{ item.role | markdownify }}</div>{% endif %}
        {% if item.authors or item.name %}<div class="record-authors markdown-content">{{ item.authors | default: item.name | markdownify }}</div>{% endif %}
        {% if item.venue %}<p class="record-venue">{{ item.venue }}{% if item.year %}, {{ item.year }}{% endif %}</p>{% endif %}
        {% if item.description %}<div class="markdown-content">{{ item.description | markdownify }}</div>{% endif %}
        <div class="record-links">
          {% for link in item.links %}
            {% if link.url and link.url != empty %}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% endif %}
          {% endfor %}
        </div>
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
  <section class="about-section about-records" aria-labelledby="technical_skills-heading">
    <h2 id="technical_skills-heading">Technical Skills</h2>
    {% if bio.technical_skills.size > 0 %}
    <ul class="record-list">
      {% for item in bio.technical_skills %}
      <li>
        <div class="record-title markdown-content">{{ item.title | markdownify }}</div>
        {% if item.date %}<p class="record-date">{{ item.date }}</p>{% endif %}
        {% if item.role %}<div class="record-role markdown-content">{{ item.role | markdownify }}</div>{% endif %}
        {% if item.authors or item.name %}<div class="record-authors markdown-content">{{ item.authors | default: item.name | markdownify }}</div>{% endif %}
        {% if item.venue %}<p class="record-venue">{{ item.venue }}{% if item.year %}, {{ item.year }}{% endif %}</p>{% endif %}
        {% if item.description %}<div class="markdown-content">{{ item.description | markdownify }}</div>{% endif %}
        <div class="record-links">
          {% for link in item.links %}
            {% if link.url and link.url != empty %}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% endif %}
          {% endfor %}
        </div>
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
</div>

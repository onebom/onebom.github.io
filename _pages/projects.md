---
layout: page
title: Project
permalink: /projects/
nav: true
nav_order: 3
---

<link rel="stylesheet" href="{{ '/assets/css/projects.css' | relative_url }}">

<div class="project-list">
  {% assign entries = site.data.about.publications | concat: site.data.about.projects %}
  {% for project in entries %}
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
      <h2>{% if project_page_url and project_page_url != empty %}<a class="project-title-link" href="{{ project_page_url | escape }}">{% endif %}{{ project.title | markdownify | remove: '<p>' | remove: '</p>' }}{% if project_page_url and project_page_url != empty %}</a>{% endif %}</h2>
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

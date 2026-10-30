---
layout: default
title: about
permalink: /
---

<link rel="stylesheet" href="{{ '/assets/css/about.css' | relative_url }}">
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
    <ul class="record-list">
      {% for item in bio.publications %}
      <li>
        <div class="record-title markdown-content">{{ item.title | markdownify }}</div>
        {% if item.date %}<p class="record-date">{{ item.date }}</p>{% endif %}
        {% if item.role %}<div class="record-role markdown-content">{{ item.role | markdownify }}</div>{% endif %}
        {% if item.authors or item.name %}<div class="record-authors markdown-content">{{ item.authors | default: item.name | markdownify }}</div>{% endif %}
        {% if item.venue %}<p class="record-venue">{{ item.venue }}{% if item.year %}, {{ item.year }}{% endif %}</p>{% endif %}
        {% if item.description %}<div class="markdown-content">{{ item.description | markdownify }}</div>{% endif %}
        {% if item.link and item.link != empty %}
        <ul class="record-link-list"><li class="markdown-content">{{ item.link | markdownify }}</li></ul>
        {% elsif item.links.size > 0 %}
        {% capture link_items %}{% assign link_separator = '' %}{% for link in item.links %}{% if link.url and link.url != empty %}{{ link_separator }}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% assign link_separator = ', ' %}{% endif %}{% endfor %}{% endcapture %}
        {% assign link_items = link_items | strip %}
        {% if link_items != empty %}<ul class="record-link-list"><li>{{ link_items }}</li></ul>{% endif %}
        {% endif %}
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="section-empty">Details to be added.</p>
    {% endif %}
  </section>
  <section class="about-section about-records" aria-labelledby="projects-heading">
    <h2 id="projects-heading">Projects</h2>
    {% if bio.projects.size > 0 %}
    <ul class="record-list">
      {% for item in bio.projects %}
      <li>
        <div class="record-title markdown-content">{{ item.title | markdownify }}</div>
        {% if item.date %}<p class="record-date">{{ item.date }}</p>{% endif %}
        {% if item.role %}<div class="record-role markdown-content">{{ item.role | markdownify }}</div>{% endif %}
        {% if item.authors or item.name %}<div class="record-authors markdown-content">{{ item.authors | default: item.name | markdownify }}</div>{% endif %}
        {% if item.venue %}<p class="record-venue">{{ item.venue }}{% if item.year %}, {{ item.year }}{% endif %}</p>{% endif %}
        {% if item.description %}<div class="markdown-content">{{ item.description | markdownify }}</div>{% endif %}
        {% if item.link and item.link != empty %}
        <ul class="record-link-list"><li class="markdown-content">{{ item.link | markdownify }}</li></ul>
        {% elsif item.links.size > 0 %}
        {% capture link_items %}{% assign link_separator = '' %}{% for link in item.links %}{% if link.url and link.url != empty %}{{ link_separator }}<a href="{{ link.url | escape }}">{{ link.label }}</a>{% assign link_separator = ', ' %}{% endif %}{% endfor %}{% endcapture %}
        {% assign link_items = link_items | strip %}
        {% if link_items != empty %}<ul class="record-link-list"><li>{{ link_items }}</li></ul>{% endif %}
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

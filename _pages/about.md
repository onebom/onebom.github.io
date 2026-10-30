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
          <a href="{{ link.url }}">{{ link.label }}</a>{% unless forloop.last %}<span aria-hidden="true">/</span>{% endunless %}
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
    <ul class="record-list">
      {% for item in bio.education %}
      <li>
        <strong>{{ item.title }}</strong>
        <ul>
          <li>{{ item.date }}</li>
          {% if item.description != empty %}<li>{{ item.description }}</li>{% endif %}
        </ul>
      </li>
      {% endfor %}
    </ul>
  </section>
  <section class="about-section about-records" aria-labelledby="publications-heading">
    <h2 id="publications-heading">Publications</h2>
    {% if bio.publications.size > 0 %}
    <ul class="record-list">
      {% for item in bio.publications %}
      <li>
        <strong>{{ item.title }}</strong>
        <ul>
          {% if item.authors %}<li>{{ item.authors }}</li>{% endif %}
          <li>{% if item.venue %}<strong>{{ item.venue }}</strong>{% endif %}{% if item.year %}, {{ item.year }}{% endif %}{% if item.url %} · <a href="{{ item.url }}">Paper</a>{% endif %}</li>
        </ul>
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="publication-empty">Publication details will be added here.</p>
    {% endif %}
  </section>
  <section class="about-section" aria-labelledby="research-heading">
    <h2 id="research-heading">Research</h2>
    <p>{{ bio.research_intro }}</p>
    <div class="research-list">
      {% for item in bio.research %}
      <article class="research-entry{% if item.highlighted %} research-highlight{% endif %}">
        <div class="research-thumbnail" aria-hidden="true"><span>{{ item.mark }}</span><small>{{ item.tag }}</small></div>
        <div class="research-details">
          <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
          <p class="research-venue">{{ item.venue }}, {{ item.year }}</p>
          <a class="research-code" href="{{ item.url }}">project / code</a>
          <p class="research-description">{{ item.description }}</p>
        </div>
      </article>
      {% endfor %}
    </div>
  </section>
  <section class="about-section" aria-labelledby="activities-heading">
    <h2 id="activities-heading">Research Activities</h2>
    {% for item in bio.activities %}
    <div class="about-timeline">
      <p class="timeline-date">{{ item.date }}</p>
      <div><h3>{{ item.title }}</h3><p class="activity-role">{{ item.role }}</p>{% if item.description != empty %}<p>{{ item.description }}</p>{% endif %}</div>
    </div>
    {% endfor %}
  </section>
  <section class="about-section" aria-labelledby="projects-heading">
    <h2 id="projects-heading">Selected Projects</h2>
    {% for item in bio.projects %}
    <div class="about-project"><h3><a href="{{ item.url }}">{{ item.title }}</a></h3><p>{{ item.description }}</p></div>
    {% endfor %}
  </section>
</div>

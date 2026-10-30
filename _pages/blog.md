---
layout: page
title: Blog
permalink: /blog/
nav: true
nav_order: 2
---

{% if site.posts.size > 0 %}
<ul class="post-list">
  {% for post in site.posts %}
  <li style="margin-bottom: 1.5rem;">
    <h2 style="margin-bottom: 0.25rem;"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p class="post-meta">{{ post.date | date: '%B %d, %Y' }}</p>
    {% if post.description %}<p>{{ post.description }}</p>{% else %}{{ post.excerpt }}{% endif %}
  </li>
  {% endfor %}
</ul>
{% else %}
<p>Research notes and updates will be shared here.</p>
{% endif %}

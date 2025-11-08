# moodle-api-python-wrapper

A lightweight Python wrapper around Moodle’s REST web services **plus** helpers for common LMS workflows: course/section/module management, H5P content creation (MCQ, Speak-the-Words, Interactive Video), and a few reporting utilities. Uses `requests` for HTTP, `SQLAlchemy`/`pandas` for DB reads, and optional Plotly figures for analytics.

> Works with Python **3.9+**.

---

## Features

* ✅ Simple REST calls to Moodle WS (`call`, `call_moodle_webservice`)
* ✅ DB helpers via SQLAlchemy for reads (e.g., categories, courses, modules)
* ✅ H5P builders/updaters (MCQ, Speak the Words, Fill-in-the-Blanks, Drag the Words, Interactive Video)
* ✅ Course/section/module create, update, move, delete
* ✅ Enrollment and role assignment helpers
* ✅ Optional analytics charts (Plotly)

---

## Installation

```bash
pip install moodle-api-python-wrapper
# Optional plotting support:
pip install "moodle-api-python-wrapper[plotting]"
```

### Runtime dependencies (core)

* numpy, pandas, requests, urllib3
* SQLAlchemy, mysql-connector-python
* pytz, scikit-learn, networkx
* pytube, xmltodict, Django (for a few helpers)

Plotting extra:

* plotly (installed via the `plotting` extra)

---

## Quickstart

### 1) Prepare credentials

Enable a **Token** + **REST** web service in Moodle (Site administration → Server → Web services).
You’ll need:

* `URL` – your Moodle base URL, e.g. `https://moodle.example.com/`
* `ENDPOINT` – typically `webservice/rest/server.php`
* `KEY` – the user’s web service token

```python
mWAParams = {
    "URL": "https://moodle.example.com/",
    "ENDPOINT": "webservice/rest/server.php",
    "KEY": "YOUR_WS_TOKEN"
}
```

### 2) (Optional) DB connection for read-only queries

The code constructs a URL with the `mysql+mysqlconnector` dialect:

```python
moodle_db = {
    "moodle": {
        "USER": "dbuser",
        "PASSWORD": "dbpass",
        "HOST": "db.example.com",
        "NAME": "moodle"
    }
}
```

### 3) Use the wrapper

```python
from typing import Optional
from moodle_api_python_wrapper import MgMoodle

m = MgMoodle(mWAParams, moodle_db)

# List all users (label=email, value=id)
resp = m.get_users_list()
print(resp["status"], resp["response"]["data"])
```

---

## Common Recipes

### Call a Moodle function directly

```python
# Example: get enrolled users in a course
courseid = 42
users = m.call(mWAParams, "core_enrol_get_enrolled_users", courseid=courseid)
```

### Read from the DB with pandas/SQLAlchemy

```python
sql = "SELECT id, shortname, fullname FROM mdl_course LIMIT 10"
out = m.get_sql_request(sql)
print(out["status"])
print(out["response"]["data"])  # JSON string with records
```

### Create course sections in bulk

```python
sections = [
    {"courseid": 42, "sectionname": "Week 1", "sectionnumber": 1},
    {"courseid": 42, "sectionname": "Week 2", "sectionnumber": 2},
]
r = m.create_course_sections(sections)
print(r["status"], r["response"]["message"])
```

### Enrol users

```python
enrolments = [
    {"roleid": 5, "userid": 123, "courseid": 42}  # roleid=5 usually "student"
]
print(m.enroll_users_moodle_course(enrolments))
```

---

## H5P Helpers

### Create an H5P MCQ quiz from a template

You need an existing **H5P template module** in Moodle (e.g., an MCQ activity). Use its `coursemoduleid` as the template.

```python
templatemoduleid = 36     # your MCQ template module id
courseid = 42
sectionid = 7
title = "Basics Quiz"

questions = [
    {"name":"Q1","grade":1,"question":"2+2=?", "correct":"4", "wrong1":"3", "wrong2":"5"},
    {"name":"Q2","grade":1,"question":"Capital of France?", "correct":"Paris", "wrong1":"Rome"},
]

r = m.create_hvp_MCQ_quiz(templatemoduleid, courseid, sectionid, title, questions)
print(r["status"], r["response"]["message"])
```

### Update an H5P Interactive Video with MCQ interactions

```python
moduleid = 1234  # the target H5P interactive video module id
interactions = [
    {
        "label": "Check 1",
        "question": "What is 3+4?",
        "answers": [{"text":"7","correct":1},{"text":"5","correct":0}],
        "start": 10, "end": 20
    }
]

m.update_hvp_videointeractions_MCQ(
    {"interactiveVideo": {"video": {"files":[]}, "assets":{"interactions":[{"action":{"params":{"answers":[]}, "metadata":{}}}]}}},
    url="https://youtu.be/VIDEO_ID",
    interactions=interactions
)
```

> Tip: Many H5P methods follow a **template → clone → transform** pattern.
> For MCQ/Speak-the-Words/etc., use the provided `update_hvp_*` helpers.

---

## Optional: Analytics (Plotly)

If you installed the plotting extra:

```python
r = m.get_courses_modules_student_grades_competency_figures([42])
payload = r["response"]["data"]  # JSON with records + HTML figure strings
```

Render the HTML strings in your frontend to display interactive charts.

---

## Configuration & Security Notes

* The code can call `urllib3.disable_warnings(InsecureRequestWarning)` if you use self-signed HTTPS. Prefer **valid TLS** in production.
* Ensure your WS token is scoped to the exact functions you need.
* Database credentials should be stored in environment variables or a secrets manager.

---

## Troubleshooting

* **`SystemError: Error calling Moodle API`**
  The WS token lacks permission, or the function name/params are wrong. Check **Site administration → Server → Web services**.

* **DB connect errors**
  Verify `mysql-connector-python` is installed and the host/port are reachable. Confirm user grants.

* **H5P template issues**
  Ensure your template module exists and its structure matches the corresponding `update_hvp_*` method you’re using.

---

## Development

```bash
# Clone
git clone https://github.com/mugalan/moodle-api-python-wrapper
cd moodle-api-python-wrapper

# Editable install (with plotting extra)
pip install -e ".[plotting]"
```

Run tests/linters as you normally would (add your preferred tooling).

---

## License

MIT © Mugalan

---

## Links

* Repository: [https://github.com/mugalan/moodle-api-python-wrapper](https://github.com/mugalan/moodle-api-python-wrapper)
* Moodle Web Services docs: [https://docs.moodle.org/dev/Web_service_API](https://docs.moodle.org/dev/Web_service_API)
* SQLAlchemy docs: [https://docs.sqlalchemy.org/](https://docs.sqlalchemy.org/)
* Requests docs: [https://requests.readthedocs.io/](https://requests.readthedocs.io/)

*Happy Moodling!*

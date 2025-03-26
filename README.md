# ExpendituresParser

This microservice is part of the [ExpenditureTracker](https://github.com/manuel-buser/ExpenditureTracker) project. Its goal is to parse various file formats (PDF, JPG, etc.) containing expenditure information, convert the data into a structured format, and store it in the ExpenditureTracker system—where a database and a user-friendly dashboard will display the parsed expenditures.

## Features

- **Multi-Format Parsing:**  
  Extracts expenditure data from various file formats (PDF, JPG, etc.).

- **Structured Data Output:**  
  Converts extracted information into a standardized JSON format suitable for database storage.

- **Microservice Integration:**  
  Designed to seamlessly integrate with the main ExpenditureTracker backend and dashboard.

- **Extensible Architecture:**  
  Easy to extend and add support for new file formats or custom extraction rules.

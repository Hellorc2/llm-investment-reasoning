from pyDatalog import pyDatalog

# Declare terms
pyDatalog.create_terms('X, success, google, apple')

# Define relationships
+google('Steve Jobs')
+apple('Steve Jobs')

# Define rule for success
success(X) <= google(X) & apple(X)  # X must be both a Google and Apple founder to be successful

# Query the database
print(success(X))  # Will output [] because no person is in both Google & Apple

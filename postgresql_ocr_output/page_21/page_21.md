## 3. PostgreSQL - Syntax  

This chapter provides a list of the PostgreSQL SQL commands, followed by the precise syntax rules for each of these commands. This set of commands is taken from the psql command- line tool. Now that you have Postgres installed, open the psql as:  

Program Files \(>\) PostgreSQL \(9.2>\) SQL Shell(psql).  

Using psql, you can generate a complete list of commands by using the \help command. For the syntax of a specific command, use the following command:  

postgres- # \help <command_name>  

## The SQL Statement  

An SQL statement is comprised of tokens where each token can represent either a keyword, identifier, quoted identifier, constant, or special character symbol. The table given below uses a simple SELECT statement to illustrate a basic, but complete, SQL statement and its components.  

<table><tr><td></td><td>SELECT</td><td>id, name</td><td>FROM</td><td>states</td></tr><tr><td>Token Type</td><td>Keyword</td><td>Identifiers</td><td>Keyword</td><td>Identifier</td></tr><tr><td>Description</td><td>Command</td><td>Id and name columns</td><td>Clause</td><td>Table name</td></tr></table>  

## PostgreSQL SQL commands  

## ABORT  

Abort the current transaction.  

ABORT [WORK | TRANSACTION ]  

## ALTER AGGREGATE  

Change the definition of an aggregate function.  

ALTER AGGREGATE name (type) RENAME TO new_name ALTER AGGREGATE name (type) OWNER TO new_owner
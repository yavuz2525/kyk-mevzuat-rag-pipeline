## ALTER INDEX  

Change the definition of an index.  

ALTER INDEX name OWNER TO new_owner  ALTER INDEX name SET TABLESPACE indexspace_name  ALTER INDEX name RENAME TO new_name  

## ALTER LANGUAGE  

Change the definition of a procedural language.  

ALTER LANGUAGE name RENAME TO new_name  

## ALTER OPERATOR  

Change the definition of an operator.  

ALTER OPERATOR name ( { lefttype | NONE }, { righttype | NONE })  OWNER TO new_owner  

## ALTER OPERATOR CLASS  

Change the definition of an operator class.  

ALTER OPERATOR CLASS name USING index_method RENAME TO new_name  ALTER OPERATOR CLASS name USING index_method OWNER TO new_owner  

## ALTER SCHEMA  

Change the definition of a schema.  

ALTER SCHEMA name RENAME TO new_name  ALTER SCHEMA name OWNER TO new_owner  

## ALTER SEQUENCE  

Change the definition of a sequence generator.  

ALTER SEQUENCE name [ INCREMENT [ BY ] increment ]  [ MINVALUE minvalue | NO MINVALUE ]  [ MAXVALUE maxvalue | NO MAXVALUE ]
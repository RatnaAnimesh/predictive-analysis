import os
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from neo4j import GraphDatabase

# Import centralized configuration
import config

class GraphBuilderHandler(FileSystemEventHandler):
    def __init__(self, driver):
        self.driver = driver

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            print(f"New file detected: {event.src_path}")
            # Add a small delay to ensure the file is fully written
            time.sleep(1) 
            self.process_file(event.src_path)

    def process_file(self, file_path):
        try:
            # Ensure file is not empty before processing
            if os.path.getsize(file_path) == 0:
                print(f"File is empty, deleting: {file_path}")
                os.remove(file_path)
                return

            df = pd.read_csv(file_path)
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    self.create_event_in_graph(session, row)
            print(f"Processed and deleted: {file_path}")
            os.remove(file_path)
        except pd.errors.EmptyDataError:
            print(f"File is empty or unreadable, deleting: {file_path}")
            os.remove(file_path)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def create_event_in_graph(self, session, row):
        # Extract data from the row, ensuring values are not NaN
        actor1_name = row.get('actor1_name') if pd.notna(row.get('actor1_name')) else None
        actor1_code = row.get('actor1_code') if pd.notna(row.get('actor1_code')) else None
        actor2_name = row.get('actor2_name') if pd.notna(row.get('actor2_name')) else None
        actor2_code = row.get('actor2_code') if pd.notna(row.get('actor2_code')) else None
        event_code = row.get('event_code') if pd.notna(row.get('event_code')) else None
        event_root_code = row.get('event_root_code') if pd.notna(row.get('event_root_code')) else None
        quad_class = row.get('quad_class') if pd.notna(row.get('quad_class')) else None
        goldstein_scale = row.get('goldstein_scale') if pd.notna(row.get('goldstein_scale')) else None
        avg_tone = row.get('avg_tone') if pd.notna(row.get('avg_tone')) else None
        event_date = row.get('sqldate') if pd.notna(row.get('sqldate')) else None

        if not actor1_name or not actor2_name or not event_code:
            return

        # Create nodes and relationships in the graph database
        session.execute_write(
            self._create_and_link_event,
            actor1_name,
            actor1_code,
            actor2_name,
            actor2_code,
            event_code,
            event_root_code,
            quad_class,
            goldstein_scale,
            avg_tone,
            event_date
        )

    @staticmethod
    def _create_and_link_event(tx, actor1_name, actor1_code, actor2_name, actor2_code, event_code, event_root_code, quad_class, goldstein_scale, avg_tone, event_date):
        # Cypher query to create nodes and relationships
        query = """
        // Merge Actor 1
        MERGE (a1:GeopoliticalActor {name: $actor1_name})
        ON CREATE SET a1.code = $actor1_code, a1.type = 'Unknown'
        ON MATCH SET a1.code = $actor1_code
        
        // Merge Actor 2
        MERGE (a2:GeopoliticalActor {name: $actor2_name})
        ON CREATE SET a2.code = $actor2_code, a2.type = 'Unknown'
        ON MATCH SET a2.code = $actor2_code
        
        // Create Event and relationships
        CREATE (e:Event {
            type: 'GeopoliticalEvent', 
            date: $event_date, 
            description: 'GDELT Event ' + $event_code + ' between ' + $actor1_name + ' and ' + $actor2_name
        })
        
        // Create relationship from Actor 1 to Actor 2
        CREATE (a1)-[r:INTERACTED_WITH {
            event_code: $event_code, 
            root_code: $event_root_code, 
            quad_class: $quad_class, 
            goldstein_scale: $goldstein_scale, 
            avg_tone: $avg_tone, 
            date: $event_date
        }]->(a2)
        
        // Link event to actors
        CREATE (e)-[:IMPACTS]->(a1)
        CREATE (e)-[:IMPACTS]->(a2)
        """
        tx.run(query, 
               actor1_name=actor1_name, actor1_code=actor1_code,
               actor2_name=actor2_name, actor2_code=actor2_code,
               event_code=event_code, event_root_code=event_root_code,
               quad_class=quad_class, goldstein_scale=goldstein_scale,
               avg_tone=avg_tone, event_date=event_date)


def main():
    print("Starting graph builder...")
    # Use credentials from config
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    event_handler = GraphBuilderHandler(driver)
    observer = Observer()
    # Use path from config
    observer.schedule(event_handler, config.PASS1_OUTPUT_DIR, recursive=False)
    observer.start()
    print(f"Watching for new files in: {config.PASS1_OUTPUT_DIR}")

    try:
        while True:
            time.sleep(5) # Can sleep for longer to reduce CPU usage
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    driver.close()
    print("Graph builder stopped.")

if __name__ == "__main__":
    main()
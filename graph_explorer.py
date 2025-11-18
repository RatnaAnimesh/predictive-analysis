import argparse
from neo4j import GraphDatabase

# Import centralized configuration
import config

def clear_database(driver):
    print("Clearing the entire graph database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")

def get_actor_interactions(driver, actor_name):
    """Prints the top 15 most recent interactions for a given actor."""
    print(f"\n--- Finding recent interactions for actor: {actor_name} ---")
    query = """
    MATCH (a:GeopoliticalActor {name: $actor_name})-[r:INTERACTED_WITH]->(b)
    RETURN b.name AS Neighbor, r.event_code AS EventCode, r.date AS Date
    ORDER BY r.date DESC
    LIMIT 15
    """
    with driver.session() as session:
        results = session.run(query, actor_name=actor_name)
        records = list(results)
        if not records:
            print(f"No interactions found for actor '{actor_name}'. Note: names are case-sensitive.")
            return
            
        print(f"Displaying up to 15 most recent interactions for '{actor_name}':")
        for record in records:
            print(f"  - Interacted with: {record['Neighbor']} (Event: {record['EventCode']}, Date: {record['Date']})")

def print_default_stats(driver):
    """Prints default summary statistics about the graph."""
    with driver.session() as session:
        # Query to count nodes
        nodes_query = "MATCH (n) RETURN count(n) AS node_count"
        nodes_result = session.run(nodes_query).single()
        node_count = nodes_result["node_count"] if nodes_result else 0
        print(f"Number of nodes in the graph: {node_count}")

        # Query to count relationships
        rels_query = "MATCH ()-[r]->() RETURN count(r) AS rel_count"
        rels_result = session.run(rels_query).single()
        rel_count = rels_result["rel_count"] if rels_result else 0
        print(f"Number of relationships in the graph: {rel_count}")

        # Example: Find some actors and their relationships
        print("\n--- Example: Top 5 Actors by number of relationships ---")
        top_actors_query = """
        MATCH (a:GeopoliticalActor)-[r:INTERACTED_WITH]->()
        RETURN a.name AS Actor, count(r) AS RelationshipCount
        ORDER BY RelationshipCount DESC
        LIMIT 5
        """
        top_actors_result = session.run(top_actors_query)
        for record in top_actors_result:
            print(f"  - {record['Actor']} (Relationships: {record['RelationshipCount']})")

def main():
    parser = argparse.ArgumentParser(description="Graph Explorer for the Geopolitical Knowledge Graph")
    parser.add_argument("--clear", action="store_true", help="Clear the entire graph database.")
    parser.add_argument("--actor", type=str, help="Name of a specific actor to query for interactions.")
    
    args = parser.parse_args()

    print(f"Connecting to graph database at {config.NEO4J_URI}...")
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    try:
        if args.clear:
            clear_database(driver)
        elif args.actor:
            get_actor_interactions(driver, args.actor)
        else:
            print_default_stats(driver)

    except Exception as e:
        print(f"Error connecting to or querying the database: {e}")
    finally:
        driver.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    main()



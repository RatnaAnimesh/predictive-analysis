from SPARQLWrapper import SPARQLWrapper, JSON
from neo4j import GraphDatabase
from tqdm import tqdm

# Import centralized configuration
import config

def get_countries_from_dbpedia():
    """Fetches a list of countries and their details from DBpedia via SPARQL."""
    sparql = SPARQLWrapper(config.DBPEDIA_URL)
    sparql.setQuery("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?country ?countryLabel ?isoCode ?capitalLabel WHERE {
          ?country a dbo:Country .
          ?country rdfs:label ?countryLabel .
          OPTIONAL { ?country dbo:isoCode ?isoCode . }
          OPTIONAL { ?country dbo:capital ?capital . ?capital rdfs:label ?capitalLabel . }
          FILTER (lang(?countryLabel) = 'en')
          FILTER (lang(?capitalLabel) = 'en')
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    countries = []
    for result in results["results"]["bindings"]:
        country_data = {
            "uri": result["country"]["value"],
            "name": result["countryLabel"]["value"],
            "iso_code": result.get("isoCode", {}).get("value"),
            "capital": result.get("capitalLabel", {}).get("value")
        }
        countries.append(country_data)
    return countries

def _create_country_node(tx, name, iso_code, capital, uri):
    """A single transaction function to merge a Country node."""
    query = (
        "MERGE (c:Country {name: $name}) "
        "ON CREATE SET c.iso_code = $iso_code, c.capital = $capital, c.uri = $uri, c.type = 'Country', c.region = 'Unknown' "
        "ON MATCH SET c.iso_code = $iso_code, c.capital = $capital, c.uri = $uri"
    )
    tx.run(query, name=name, iso_code=iso_code, capital=capital, uri=uri)

def populate_graph_with_countries(driver, countries):
    """Populates the graph with country nodes from the DBpedia list."""
    with driver.session() as session:
        for country in tqdm(countries, desc="Populating countries"):
            session.execute_write(
                _create_country_node,
                country["name"],
                country["iso_code"],
                country["capital"],
                country["uri"]
            )

def main():
    print(f"Fetching country data from DBpedia at {config.DBPEDIA_URL}...")
    try:
        countries = get_countries_from_dbpedia()
        print(f"Found {len(countries)} countries.")
    except Exception as e:
        print(f"Error fetching data from DBpedia: {e}")
        return

    print(f"Connecting to graph database at {config.NEO4J_URI}...")
    # Use credentials from config
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    try:
        print("Populating graph with countries...")
        populate_graph_with_countries(driver, countries)
        print("Successfully populated graph with country data.")
    except Exception as e:
        print(f"Error populating graph: {e}")
    finally:
        driver.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()


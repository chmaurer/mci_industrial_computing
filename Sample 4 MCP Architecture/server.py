from fastmcp import FastMCP

# Initialize the FastMCP server instance
mcp = FastMCP("Finance_Tools")

@mcp.tool()
def calculate_tax(subtotal: float, state: str) -> float:
    """Calculates sales tax based on the state code."""
    rates = {"NY": 0.08875, "CA": 0.0825, "TX": 0.0625}
    return subtotal * rates.get(state.upper(), 0.0)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)

#run this on your server
#sudo docker run --name=finance_mcp --rm --net=host 192.168.1.60:8083/repository/dockerrepo/finance_mcp:1.4
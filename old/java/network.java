//Christopher de la Iglesia

public class network {

    ArrayList<node> nodes;

    public network(int num, int maxTrigger, double levelp) {
	nodes = new ArrayList<nodes>();
	int level = levelp*maxTrigger;
	Random rng = new Random();
	for(int i = 0; i < num; i++) {
	    nodes.add(new node(rng.nextInt(maxTrigger),level,i);
	}
    }

    public int run(int src, int dst) {
	int time = 0;
	while(!dst.fired) {
	    src.activate();
	    time++;
	} 
	return time;
    }

}

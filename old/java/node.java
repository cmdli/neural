//Christopher de la Iglesia

class node {
    
    ArrayList<node> subset;

    int trigger;
    int level;
    int id;
    boolean fired;

    public node(int ntrigger, int nid) {
	this(ntrigger,0,nid);
    }

    public node(int ntrigger, int nlevel, int nid) {
	subset = new ArrayList<node>();
	trigger = ntrigger;
	level = nlevel;
	id = nid;
	fired = false;
    }

    public void activate() {
	level++;
	if(level >= trigger) {
	    fire();
	    level = 0;
	    fired = true;
	}
    }

    public void add(ArrayList<node> allNodes) {
	subset.add(
		   allNodes.get(
				new Random().nextInt(allNodes.size())
				)
		   );
    }

    public void remove() {
	subset.remove(
		      new Random().nextInt
		      (
		          subset.size()
		      )
	);
    }

    private void fire() {
	for(node n : subset) {
	    n.activate();
	}
    }

}

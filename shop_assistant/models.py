from django.db import models


class SearchLog(models.Model):
    """Log user searches for analytics."""
    category = models.CharField(max_length=100, blank=True, null=True)
    keyword = models.CharField(max_length=200, blank=True, null=True)
    budget = models.FloatField(null=True, blank=True)
    results_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Search: {self.keyword or self.category} @ {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']


class FavoriteProduct(models.Model):
    """Track favorited products (session-based)."""
    product_name = models.CharField(max_length=200)
    brand = models.CharField(max_length=100)
    session_key = models.CharField(max_length=40)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.product_name} by {self.brand}"
